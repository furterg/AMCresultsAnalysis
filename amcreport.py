#!/usr/bin/env python3
import configparser
import glob
import os
import sqlite3
import sys
import json
import datetime
import shutil
import subprocess
import platform
from openai import OpenAI
import pandas as pd
import seaborn as sns
from scipy import stats
from report import generate_pdf_report, plot_charts
from icecream import install, ic


def get_dictionary(dictionary: str) -> dict:
    """
    Get the definitions from the definitions.json file
    :return: a dictionary of definitions
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dictionary + ".json")

    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {str(e)}")
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
    return data


def get_settings(filename: str) -> list:
    """
    Read the settings from a configuration file, or create a new one if it doesn't exist.

    :param filename: the name of the configuration file:
    :return: values of a dictionary of the settings
    """

    if not os.path.isfile(filename):
        create_file = input(f"The file {filename} doesn't exist. Do you want to create it? (y/n): ")
        default_dir = '/path/to/project/dir'
        if create_file.lower() == 'y':
            look_for_path = input(
                'Do you want to automatically search for the "Projets-QCM" directory? (y/n)')
            if look_for_path.lower() == 'y':
                home_dir = os.path.expanduser("~")
                projects_dir = None
                for dirpath, dirnames, filenames in os.walk(home_dir):
                    if "Projets-QCM" in dirnames:
                        projects_dir = os.path.join(dirpath, "Projets-QCM")
                        break
                if projects_dir:
                    print(f"The full path to the 'Projets-QCM' directory is: {projects_dir}")
                    default_dir = projects_dir
                else:
                    print("Could not find the 'Projets-QCM' directory.")
            config = configparser.ConfigParser()
            config['DEFAULT'] = {
                'projects_dir': default_dir,
            }
            with open(filename, 'w') as configfile:
                config.write(configfile)
        else:
            raise ValueError(f"The file {filename} was not created.")
    else:
        config = configparser.ConfigParser()
        config.read(filename)
        settings = {
            'projects_dir': config.get('DEFAULT', 'projects_dir'),
            'student_threshold': int(config.get('DEFAULT', 'student_threshold')),
            'company_name': config.get('DEFAULT', 'company_name'),
            'company_url': config.get('DEFAULT', 'company_url'),
        }
        return list(settings.values())


def get_project_directories(path: str) -> str:
    """
    - Get the list of project directories
    - Presents the list to the user
    - Get the user's selection
    :param path: path to the Projets-QCM directory
    :return: path to the project selected by the user
    """
    # list subdirectories and sorts them
    if not os.path.exists(path):
        print(f"The path {path} does not exist.")
        sys.exit(1)

    subdirectories: list = next(os.walk(path))[1]
    subdirectories.remove('_Archive')
    subdirectories.sort()

    if len(sys.argv) > 1 and sys.argv[1] in subdirectories:
        return os.path.join(path, sys.argv[1])
    while True:
        # display numbered list of subdirectories
        print("Here's a list of current projects:")
        for i, subdirectory in enumerate(subdirectories):
            print(f"{i + 1}. {subdirectory}")

        # prompt user to select a subdirectory
        selection: str = input("Enter the number of the project you'd like to select: ")

        # validate user input
        while not selection.isdigit() \
                or int(selection) not in range(0, len(subdirectories) + 1):
            selection = input(
                "Invalid input. Enter the number of the project you'd like to select \
                (type 0 for list): ")

        # If user input is 0, then print the list again
        if selection == '0':
            continue

        # store the path to the selected project
        selected_path: str = os.path.join(path, subdirectories[int(selection) - 1])
        return selected_path


def get_tables(db):
    """
    Get the scoring data by querying 'scoring.sqlite' from the project directory:

    * pd_mark contains the marks obtained by the students:
        - student: student id in the database
        - total: total points obtained by the student
        - max: maximum obtainable mark
        - mark: mark obtained by the student
    * pd_score are the scores obtained by each student for each question and the status \
    of the question:
        - student: student id in the database
        - question: question id in the database
        - score: score obtained by the student
        - max: maximum score for the question
        - replied: question is replied (True/False)
        - canceled: question is canceled (True/False) - optional
        - flood: question is floored (True/False) - optional
        - empty: question is empty (True/False) - optional
        - error: answer is incoherent (True/False) - optional
    * pd_variables related to the exam (name and value). Notable variables are:
        - darkness_threshold: lower darkness limit to consider a box as ticked
        - darkness_threshold: upper darkness limit to consider a box as ticked
        - mark_max: the maximum mark for the exam
    * pd_question is the list of all individual questions with additional data (some columns \
    optional):
        - question: question id in the database
        - title: name of the question in the exam catalog (i.e. Q008)
        - canceled: number of times the question has been canceled (skipped and not counted)
        - correct: number of times the question has been answered correctly
        - empty: number of times the question has been left blank
        - error: number of times the question has been incoherent (several boxes ticked)
        - floored: number of times the question has been floored (for multiple answers questions)
        - max: sum of maximum score for the question
        - replied: number of times the question has been replied
        - score: sum of obtained scores for the question
        - presented: number of times the question has been presented
        - difficulty: difficulty of the question (sum of scores / sum of max WHERE why != 'C')
    * pd_answer contains all the questions and answers with indication of correct answers:
        - question: question id in the database
        - answer: answer number for the questions (1=A, 2=B, ...)
        - correct: 1 if the answer is correct, 0 otherwise
        - strategy: additional scoring indication. A 1 here may indicate another correct answer


    :param db: path to the database
    :return: a tuple if pandas dataframes (pd_mark, pd_score, pd_variables, pd_question, pd_answer)
    """
    # create a connection to the database
    conn = sqlite3.connect(db)

    pd_mark = pd.read_sql_query("SELECT * FROM scoring_mark", conn)
    pd_score = pd.read_sql_query(f"""
    SELECT ss.student, ss.question, st.title, ss.score, ss.why, ss.max
    FROM scoring_score ss
    JOIN scoring_title st ON ss.question = st.question
    WHERE ss.question > {student_code_length}""", conn)
    pd_answer = pd.read_sql_query(f"""SELECT DISTINCT question, answer, correct, strategy
                                  FROM scoring_answer 
                                  WHERE question > {student_code_length}""", conn)
    pd_variables = pd.read_sql_query("SELECT * FROM scoring_variables", conn, index_col='name')
    # Get the list of question IDs for non-indicative questions
    pd_indicative = pd.read_sql_query("""SELECT DISTINCT question 
                                        FROM scoring_question 
                                        WHERE indicative = 1""", conn)
    # close the database connection
    conn.close()

    if pd_mark.empty:
        print("Error: No mark has been recorded in the database")
        sys.exit()

    # Clean the scores to keep track of Cancelled (C), Floored (P), Empty (V) and Error (E)
    # questions
    why = pd.get_dummies(pd_score['why'])
    pd_score = pd.concat([pd_score, why], axis=1)
    pd_score.drop('why', axis=1, inplace=True)
    pd_score.rename(columns={'': 'replied', 'C': 'cancelled', 'P': 'floored', 'V': 'empty',
                             'E': 'error'}, inplace=True)
    pd_score['correct'] = pd_score.apply(lambda row: 1 if row['score'] == row['max'] else 0, axis=1)

    # Create pd_question as a pivot table of pd_scores. It now contains the following columns:
    # question title  cancelled  correct  empty error floored  max  replied  score (some columns
    # are optional)
    pd_question = pd_score.pivot_table(index=['question', 'title'],
                                       values=pd_score.columns[3:],
                                       aggfunc='sum',
                                       fill_value=0).reset_index()
    # Remove rows in dataframes where question is present in pd_indicative
    pd_question = pd_question.drop(
        pd_question[pd_question['question'].isin(pd_indicative['question'])].index)
    pd_answer = pd_answer.drop(
        pd_answer[pd_answer['question'].isin(pd_indicative['question'])].index)
    pd_score = pd_score.drop(pd_score[pd_score['question'].isin(pd_indicative['question'])].index)

    # Get the list of columns to calculate the number of times a question has been presented.
    cols_for_pres = [col for col in ['cancelled', 'empty', 'replied', 'error', 'floored']
                     if col in pd_question.columns]
    pd_question['presented'] = pd_question[cols_for_pres].sum(axis=1)
    # Combined 'replied' and 'floored' as 'replied' otherwise the number of presented is wrong
    # Check if there is a column 'floored' in 'pd_question'
    if 'floored' in pd_question.columns:
        # Create a new column 'replied_new' by summing 'replied' and 'floored'
        pd_question['replied_new'] = pd_question['replied'] + pd_question['floored']
        # Drop the original 'replied' column
        pd_question = pd_question.drop('replied', axis=1)
        # Rename 'replied_new' to 'replied'
        pd_question = pd_question.rename(columns={'replied_new': 'replied'})

    # Get the list of columns for calculate the number of times a question has been replied or
    # left empty...
    cols_for_diff = [col for col in ['floored', 'empty', 'replied', 'error']
                     if col in pd_question.columns]
    # Calculate the difficulty of each question
    pd_question['difficulty'] = pd_question['correct'] / pd_question[cols_for_diff].sum(axis=1)
    # Now the columns are: ['question', 'title', 'cancelled', 'correct', 'empty', 'error', 'max',
    # 'replied', 'score', 'presented', 'difficulty'] - some columns are optional

    # Apply specific operations to pd_answer before returning it
    pd_answer['correct'] = pd_answer.apply(lambda x: 1 if (x['correct'] == 1) else 0, axis=1)
    return pd_mark, pd_score, pd_variables, pd_question, pd_answer, pd_indicative


def ticked(row):
    """
    Define if an answer box has been ticked by looking at the darkness of the box compared to the \
    threshold.
    To be used with the capture dataframe to determine if an answer box has been ticked.
    :param row:
    :return: 1 or 0
    """
    # Get thresholds to calculate ticked answers and get the item analysis
    darkness_bottom = float(variables_df.loc['darkness_threshold']['value'])
    darkness_top = float(variables_df.loc['darkness_threshold_up']['value'])

    # If the box has been manually (un-)ticked, set 'ticked' to 1 (ticked) or 0 (un-ticked).
    if row['manual'] != -1:
        return row['manual']
    # If the box darkness is within the threshold => 'ticked' = 1
    elif row['total'] * darkness_bottom < row['black'] <= row['total'] * darkness_top:
        return 1
    else:
        return 0


def get_capture_table(db):
    """
    Get the capture data by querying 'capture.sqlite' from the project directory:
    FRom this dataframe, we can determine if an answer box has been ticked based on the \
    darkness of the box compared to the threshold.
    * pd_capture contains all the questions and answers with indication of correct answers:
        - question: question id in the database
        - answer: answer number for the questions (1=A, 2=B, ...)
        - correct: 1 if the answer is correct, 0 otherwise

    :param db: path to the database
    :return capture_table and questions and answers summary:
    """
    # create a connection to the database
    conn = sqlite3.connect(db)

    pd_capture = pd.read_sql_query(f"""
    SELECT student, id_a AS 'question', id_b AS 'answer', total, black, manual 
    FROM capture_zone 
    WHERE type = 4 AND id_a > {student_code_length}""", conn)

    # close the database connection
    conn.close()

    # Remove rows in pd_capture where question is present in pd_indicative
    pd_capture = pd_capture.drop(
        pd_capture[pd_capture['question'].isin(indicative_df['question'])].index)

    # Apply specific operations to dataframes before returning them
    # pd_capture
    pd_capture['ticked'] = pd_capture.apply(ticked, axis=1)

    # pd_items
    pd_items = pd_capture.groupby(['question', 'answer'])['ticked'].sum().reset_index().sort_values(
        by=['question', 'answer'])
    pd_items['correct'] = pd_items.apply(lambda row: answer_df.loc[
        (answer_df['question'] == row['question'])
        & (answer_df['answer'] == row['answer']), 'correct'].values[0], axis=1)
    pd_items = pd_items.merge(question_df[['question', 'title']],
                              left_on='question', right_on='question')
    pd_items = pd_items[['question', 'title', 'answer', 'correct', 'ticked']].sort_values(
        by=['title', 'answer']).reset_index(drop=True)

    return pd_capture, pd_items


def general_stats():
    """
    Compute the general statistics of the examination
    Create a dictionary with the statistics

    :return: dataframe of the statistics
    """
    # compute the statistics
    number_of_examinees: int = mark_df['student'].nunique()
    number_of_questions: int = question_df['title'].nunique()
    max_possible_score = float(variables_df['value']['mark_max'])
    min_achieved_score = mark_df['mark'].min()
    max_achieved_score = mark_df['mark'].max()
    mean_score = mark_df['mark'].mean()
    median_score = mark_df['mark'].median()
    mode_score = mark_df['mark'].mode().iloc[0]
    std_score = mark_df['mark'].std()
    var_score = mark_df['mark'].var()
    sem_score = stats.sem(mark_df['mark'])
    sem_measurement = std_score / (number_of_examinees ** 0.5)
    skewness = mark_df['mark'].skew()
    kurtosis = mark_df['mark'].kurtosis()
#     alpha = pg.cronbach_alpha(data=(score_df.select_dtypes(include=['int64', 'float64'])
#                                     if 'cancelled' not in score_df.columns
#                                     else score_df[score_df['cancelled'] == 0].select_dtypes(
#         include=['int64', 'float64'])),
#                               items='question',
#                               scores='score')

    # create a dictionary to store the statistics
    stats_dict = {
        'Number of examinees': number_of_examinees,
        'Number of questions': number_of_questions,
        'Maximum possible mark': max_possible_score,
        'Minimum achieved mark': min_achieved_score,
        'Maximum achieved mark': max_achieved_score,
        'Mean': mean_score,
        'Median': median_score,
        'Mode': mode_score,
        'Standard deviation': std_score,
        'Variance': var_score,
        'Standard error of mean': sem_score,
        'Standard error of measurement': sem_measurement,
        'Skew': skewness,
        'Kurtosis': kurtosis,
    }

    # create a Pandas DataFrame from the dictionary
    pd_stats = pd.DataFrame.from_dict(stats_dict, orient='index', columns=['Value'])

    return pd_stats


def questions_discrimination():
    """
    Calculate the discrimination index for each question.
    Add a column 'discrimination' to the dataframe 'question_df' with the index for each question
    :return: a list of discrimination indices to be added as a column to question_df
    """
    # Merge questions scores and students mark, bottom quantile
    bottom_merged_df = pd.merge(bottom_27_df,
                                (score_df.select_dtypes(include=['int64', 'float64'])
                                 if 'cancelled' not in score_df.columns
                                 else score_df[score_df['cancelled'] == 0].
                                 select_dtypes(include=['int64', 'float64'])),
                                on=['student'], how="inner", validate="many_to_many")

    # Merge questions scores and students mark, top quantile
    top_merged_df = pd.merge(top_27_df,
                             (score_df.select_dtypes(include=['int64', 'float64'])
                              if 'cancelled' not in score_df.columns
                              else score_df[score_df['cancelled'] == 0].
                              select_dtypes(include=['int64', 'float64'])),
                             on=['student'], how="inner", validate="many_to_many")

    # Group by question and answer, and calculate the mean mark for each group
    top_mean_df = top_merged_df.groupby(['question', 'student']).mean()
    bottom_mean_df = bottom_merged_df.groupby(['question', 'student']).mean()

    # Calculate the discrimination index for each question
    discrimination = []  # Create a list to store the results
    nb_in_groups = round(len(mark_df) * 0.27)
    for question in top_mean_df.index.levels[0]:
        discr_index = (len(top_mean_df.loc[question][
                               top_mean_df.loc[question]['score'] == top_mean_df.loc[question][
                                   'score'].max()])
                       - len(bottom_mean_df.loc[question][
                                 bottom_mean_df.loc[question]['score'] ==
                                 bottom_mean_df.loc[question]['score'].max()])) / nb_in_groups
        discrimination.append(discr_index)  # Add the result to the list
    return discrimination


def items_discrimination():
    """
    Calculate the discrimination index for each answer.
    Add a column 'discrimination' to the dataframe 'items_df' with the index for each choice
    :return: a list of discrimination indices to be added as a column to items_df
    """
    # Merge questions scores and students mark, bottom quantile
    bottom_merged_df = bottom_27_df.merge(capture_df[['student', 'question', 'answer', 'ticked']],
                                          on='student', how='left')

    # Merge questions scores and students mark, top quantile
    top_merged_df = top_27_df.merge(capture_df[['student', 'question', 'answer', 'ticked']],
                                    on='student', how='left')

    # Group by question and answer, and calculate the mean mark for each group
    top_sum_df = top_merged_df[['question', 'answer', 'ticked']].groupby(
        ['question', 'answer']).sum()
    bottom_sum_df = bottom_merged_df[['question', 'answer', 'ticked']].groupby(
        ['question', 'answer']).sum()

    # Calculate the discrimination index for each question
    # Create a dictionary to store the results
    discrimination = {'question': [], 'answer': [], 'discrimination': []}
    nb_in_groups = round(len(mark_df) * 0.27)
    for question in top_sum_df.index.levels[0]:
        for answer in top_sum_df.loc[question].index:
            discr_index = (top_sum_df.loc[question, answer]['ticked']
                           - bottom_sum_df.loc[question, answer]['ticked']) \
                          / nb_in_groups
            discrimination['question'].append(question)
            discrimination['answer'].append(answer)
            discrimination['discrimination'].append(discr_index)

    pd_discrimination = pd.DataFrame.from_dict(discrimination, orient='columns')
    return pd_discrimination


def get_item_correlation():
    """
    Calculate the item correlation for each question.
    :return: a dictionary of item correlations with questions as keys
    """
    if 'cancelled' in score_df.columns:
        merged_df = pd.merge(score_df[score_df['cancelled'] == False], mark_df, on='student',
                             how="inner", validate="many_to_many")
    else:
        merged_df = pd.merge(score_df, mark_df, on='student', how="inner", validate="many_to_many")
    item_corr = {}
    questions = merged_df['title'].unique()
    for question in questions:
        item_scores = merged_df.loc[merged_df['title'] == question, 'correct']
        total_scores = merged_df.loc[merged_df['title'] == question, 'mark']
        correlation = stats.pointbiserialr(item_scores, total_scores)
        item_corr[question] = correlation[0]
    item_corr_df = pd.DataFrame.from_dict(item_corr, orient='index', columns=['correlation'])
    return item_corr_df


def get_outcome_correlation():
    """
    Calculate the outcome correlation for each outcome of each question.
    :return: a dataframe of outcome correlations
    """
    if 'cancelled' in score_df.columns:
        merged_df = pd.merge(capture_df, score_df[['student', 'question', 'cancelled']],
                             on=['student', 'question'], how="inner", validate="many_to_many")
        merged_df = merged_df[merged_df['cancelled'] == False].merge(mark_df[['student', 'mark']],
                                                                     on='student')
    else:
        merged_df = capture_df.merge(mark_df[['student', 'mark']], on='student')
    outcome_corr = {'question': [], 'answer': [], 'correlation': []}
    questions = merged_df['question'].unique()
    for question in questions:
        answers = merged_df['answer'][merged_df['question'] == question].unique()
        for answer in answers:
            item_scores = merged_df.loc[
                (merged_df['question'] == question) & (merged_df['answer'] == answer), 'ticked']
            total_scores = merged_df.loc[
                (merged_df['question'] == question) & (merged_df['answer'] == answer), 'mark']
            correlation = stats.pointbiserialr(item_scores.values, total_scores.values)
            outcome_corr['question'].append(question)
            outcome_corr['answer'].append(answer)
            outcome_corr['correlation'].append(correlation[0])
    outcome_corr_df = pd.DataFrame.from_dict(outcome_corr)
    return outcome_corr_df


def get_gpt_response(text):
    """
    Generates text response based on the provided prompt using OpenAI's GPT-3.5 model.
    :param text: the prompt to be sent to OpenAI
    :return: the response
    :rtype: str
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=text,
            temperature=temp,
        )
        content = response.choices[0].message.content
    except Exception as e:
        content = 'Error: ' + str(e)
    return content


def init_gpt_dialogue():
    """
    Initializes a dialogue with the GPT-3.5 model to analyze the question dataframe and explain \
    the least and most performing questions.
    - Set the system prompt to shape the behaviour of ChatGPT.
    - Ask for simple statistics analysis by provinding a dataframe of the statistics.

    TODO:
        - use tabulate to shape the dataframe to a better table.
    :return: a list of prompts sent to and received from OpenAI as dictionaries.
    :rtype: list of dict
    """
    # Use OpenAI API to analyse the question dataframe and explain the least and most performing
    # questions
    table = (stats_df.reset_index(names=['Element', 'Value']).iloc[
        [2, 3, 4, 5, 6, 7, 8, 12, 13]])
    table['Value'] = table['Value'].apply(pd.to_numeric, errors='coerce')
    prompt = [{'role': 'system', 'content': stats_prompt},
              {'role': 'user',
               'content': f"Summarise the following statistics so that they are easy to "
                          f"understand. Round the numbers in your answer and give a summary at"
                          f" the end:\n{table}"}]
    response = get_gpt_response(prompt)
    prompt.append({'role': 'assistant', 'content': response})
    return prompt


def get_list_questions(qlist) -> str:
    return ', '.join(qlist[:-1]) + ' and ' + qlist[-1]


def get_blurb():
    """
    Generate a first level of analysis on the performance of the questions. This text can either \
    be used as is in the report or passed to ChatGPT for a better wording.
    :return: a string of text describing the data and how to improve the exam questions.
    """
    intro = "According to the data collected, the following questions should probably be reviewed:\n"
    blb = ''
    if ('cancelled' in question_df.columns) \
            and (question_df[question_df['cancelled'] > question_df['presented'] * 0.8][
                     'title'].values.size > 0):
        top_cancelled = question_df[question_df['cancelled']
                                    > question_df['presented'] / 1.2].sort_values('title')[
            'title'].values
        if len(top_cancelled) > 1:
            blb += (f"- Questions  {', '.join(top_cancelled[:-1]) + ' and ' + top_cancelled[-1]} "
                    f"have been cancelled more than 80% of the time.\n")
        else:
            blb += f"- Question {top_cancelled[0]} has been cancelled more than 80% of the time.\n"
    if ('empty' in question_df.columns) \
            and (question_df[question_df['empty'] > question_df['presented'] / 1.2].count()[
                     'title'] > 0):
        top_empty = question_df[question_df['empty']
                                > question_df['presented'] / 1.2].sort_values('title')[
            'title'].values
        if len(top_empty) > 1:
            qlist: str = get_list_questions(top_empty)
            blb += f"- Questions {qlist} have been empty more than 80% of the time.\n"
        else:
            blb += f"- Question {top_empty[0]} has been empty more than 80% of the time.\n"
    if ('discrimination' in question_df.columns) \
            and (question_df[question_df['discrimination'] < 0]['title'].values.size > 0):
        negative_discrimination = \
            question_df[question_df['discrimination'] < 0].sort_values('title')['title'].values
        if len(negative_discrimination) > 1:
            qlist: str = get_list_questions(negative_discrimination)
            blb += (f"- Questions {qlist} have a negative discrimination, meaning that there is a "
                    f"possibility of an error in the questions (incorrect outcome indicated as "
                    f"correct).\n")
        else:
            blb += (f"- Question {negative_discrimination[0]} has a negative discrimination, "
                    f"meaning that there is a possibility of an error in the question (incorrect "
                    f"outcome indicated as correct).\n")
    if items_df[items_df['ticked'] == 0]['title'].values.size > 0:
        not_ticked: list = items_df[items_df['ticked'] == 0]['title'].unique()
        not_ticked.sort()
        if len(not_ticked) > 1:
            qlist: str = get_list_questions(not_ticked)
            blb += f"- Questions {qlist} have distractors that have never been chosen.\n"
        else:
            blb += f"- Question {not_ticked[0]} has distractors that have never been chosen.\n"
    if len(blb) > 0:
        blb = intro + blb
    else:
        blb = ("According to the data collected, there are no questions to review based on their "
               "performance.")
    return blb


def get_student_code_length(db: str) -> int:
    """
    Get the number of student code boxes
    :param db: path to the database
    :return: Length of the student code as an integer
    """
    # create a connection to the database
    conn = sqlite3.connect(db)

    # get the number boxes for the student code, so we can only query the real questions
    scl: int = pd.read_sql_query("""SELECT COUNT(*) FROM scoring_title 
                                 WHERE title LIKE '%student.number%';""",
                                 conn).iloc[0].iloc[0]
    # close the database connection
    conn.close()
    return scl


def print_dataframes():
    """
    Print the dataframes
    """
    print(f"\nGeneral Statistics:\n{stats_df}")
    print(f"\nList of questions (question_df):\n{question_df.head()}")
    print(f"\nList of answers (answer_df):\n{answer_df.head()}")
    print(f"\nList of items (items_df):\n{items_df.sort_values(by='title').head()}")
    print(f"\nList of variables (variables_df):\n{variables_df}")
    print(f"\nList of capture (capture_df):\n{capture_df.head()}")
    print(f"\nList of mark (mark_df):\n{mark_df.head()}")
    print(f"\nList of score (score_df):\n{score_df.head()}")
    # print(f"\nitems discrimination: \n{items_discrimination()}")


def get_correction_text(df: pd.DataFrame) -> str:
    """
    Generate a paragraphe of text fron the capture data to explain the number of boxes ticked or
    unticked during the marking process.

    :param df: Dataframe with the capture data
    :type df: pd.Dataframe
    :return: a paragraph of text to display in the report
    :rtype: str
    """
    tres: int = 180
    nb_box_filled = df[(capture_df['manual'] == 1) & (capture_df['black'] < tres)]['student'].count()
    nb_box_emptied = df[(capture_df['manual'] == 0) & (capture_df['black'] > tres)]['student'].count()
    nb_box_untouched = df[(capture_df['manual'] == -1)]['student'].count()
    nb_box_total = df['student'].count()
    nb_box_untouched += df[(capture_df['manual'] == 1) & (capture_df['black'] > tres)]['student'].count()
    nb_box_untouched += df[(capture_df['manual'] == 0) & (capture_df['black'] < tres)]['student'].count()
    pc_filled: str = f"{nb_box_filled * 100 / nb_box_total:.2f}%"
    pc_emptied: str = f"{nb_box_emptied * 100 / nb_box_total:.2f}%"
    pc_untouched: str = f"{nb_box_untouched * 100 / nb_box_total:.2f}%"
    txt: str = (f"This examination is comprised of {nb_box_total} boxes in total. During the "
                f"marking process, {nb_box_filled} ({pc_filled}) have been manually filled "
                f"(ticked), {nb_box_emptied} ({pc_emptied}) have been manually emptied (un-ticked) "
                f"and {nb_box_untouched} ({pc_untouched}) have not been changed.")
    return txt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    install()
    ic.disable()
    debug: int = 0  # Set to 1 for debugging, meaning not using OpenAI
    sns.set_theme()
    sns.set_style('darkgrid')
    sns.set_style()
    sns.color_palette("tab10")
    # colour palette (red, green, blue, text color)
    colour_palette = {'heading_1': (23, 55, 83, 255),
                      'heading_2': (109, 174, 219, 55),
                      'heading_3': (40, 146, 215, 55),
                      'white': (255, 255, 255, 0),
                      'yellow': (251, 215, 114, 0),
                      'red': (238, 72, 82, 0),
                      'green': (166, 221, 182, 0),
                      'grey': (230, 230, 230, 0),
                      'blue': (84, 153, 242, 0),
                      }
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_filename: str = 'settings.conf'
    config_filename = os.path.join(script_dir, config_filename)
    # Get some directory information
    # Get the dir name to check if it matches 'Projets-QCM'
    current_dir_name: str = os.path.basename(os.getcwd())
    # Get the full path in case it matches
    current_full_path: str = os.path.dirname(os.getcwd()) + '/' + current_dir_name
    today = datetime.datetime.now().strftime('%d/%m/%Y')

    # OpenAI settings
    client = OpenAI()
    temp = 0.1
    stats_prompt = """You are a Data Scientist, specialised in the Classical Test Theory. 
    Give a short qualitative  explanation about the overall exam results. 
    Don't go into technical details, focus on meaning.
    Don't give definitions of the elements, just explain what they mean in the current context.
    Don't mention the Classical Test Theory in your reply.
    Don't introduce your answer."""
    dialogue = []  # used to store the conversation with ChatGPT

    blurb: str = ''
    directory_path, student_threshold, company_name, company_url = get_settings(
        config_filename)
    # Get the Project directory and questions file paths
    directory_path = os.path.expanduser("~") + directory_path
    amcProject: str = get_project_directories(directory_path)
    scoring_path: str = amcProject + '/data/scoring.sqlite'
    capture_path: str = amcProject + '/data/capture.sqlite'
    amcProject_name: str = glob.glob(amcProject, recursive=False)[0].split('/')[-1]
    # source_file_path: list[str] = glob.glob(amcProject + '/*source*.tex', recursive=True)

    question_data_columns: list = ['presented', 'cancelled', 'replied', 'correct', 'empty',
                                   'error', ]
    question_analysis_columns: list = ['difficulty', 'discrimination', 'correlation', ]
    outcome_data_columns: list = ['answer', 'correct', 'ticked', 'discrimination', ]

    definitions: dict = get_dictionary('definitions')
    findings: dict = get_dictionary('findings')
    # Issue an error and terminate if the scoring database does not exist
    if not os.path.exists(scoring_path):
        print("Error: the database does not exist!")
        sys.exit(1)  # terminate the program with an error code

    student_code_length: int = get_student_code_length(scoring_path)

    # get the data from the databases
    mark_df, score_df, variables_df, question_df, answer_df, indicative_df = get_tables(
        scoring_path)
    capture_df, items_df = get_capture_table(capture_path)
    # display general statistics about the exam
    stats_df = general_stats()

    # Get item and outcome discrimination if the number of examinees is greater than 99
    if stats_df.loc['Number of examinees'].iloc[0] > student_threshold:
        # Create two student dataframes based on the quantile values. They should have the same
        # number of students This should probably be done in a smarter way, outside, in order to
        # be used for item discrimination.
        top_27_df = mark_df.sort_values(by=['mark'], ascending=False).head(
            round(len(mark_df) * 0.27))
        bottom_27_df = mark_df.sort_values(by=['mark'], ascending=False).tail(
            round(len(mark_df) * 0.27))

        question_df['discrimination'] = questions_discrimination()
        items_discr = items_discrimination()
        items_df = items_df.merge(items_discr[['question', 'answer', 'discrimination']],
                                  on=['question', 'answer'])

    # Get item (question) correlation
    item_correlation = get_item_correlation()
    question_df['correlation'] = question_df['title'].apply(
        lambda row: item_correlation.loc[row]['correlation'])

    # Get outcome (answer) correlation
    outcome_correlation = get_outcome_correlation()
    items_df = items_df.merge(outcome_correlation, on=['question', 'answer'])

    # print_dataframes()
    if debug == 0:
        dialogue += init_gpt_dialogue()
        blurb += dialogue[-1]['content'] + '\n\n'
    # Generate the report
    blurb += get_blurb()
    correction_text: str = get_correction_text(capture_df)
    report_params = {
        'project_name': amcProject_name,
        'project_path': amcProject,
        'questions': question_df,
        'items': items_df,
        'stats': stats_df,
        'threshold': student_threshold,
        'marks': mark_df,
        'definitions': definitions,
        'findings': findings,
        'palette': colour_palette,
        'blurb': blurb,
        'company_name': company_name,
        'company_url': company_url,
        'correction': correction_text,
    }
    plot_charts(report_params)
    report_url: str = generate_pdf_report(report_params)
    # Open the report
    if platform.system() == 'Darwin':  # macOS
        subprocess.call(('open', report_url))
    elif platform.system() == 'Windows':  # Windows
        os.startfile(report_url)
    else:  # linux variants
        if shutil.which("zathura") is not None:
            subprocess.Popen(['zathura', report_url], start_new_session=True, stderr=subprocess.DEVNULL)
        else:
            subprocess.call(('xdg-open', report_url))

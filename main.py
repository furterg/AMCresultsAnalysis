# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import configparser
import glob
import openai
import os
import re
import sqlite3
import sys
import json
from fpdf import FPDF
import datetime

import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy import stats
from report import generate_pdf_report, plot_charts

from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode
from pylatexenc.latex2text import LatexNodes2Text

# Try to get API KEY from ENV
openai.api_key = os.getenv('OPENAI_API_KEY')

sns.set_theme()
sns.set_style('darkgrid')
sns.set_style()
sns.color_palette("tab10")
report_author = 'Gregory Furter'

config_filename = 'settings.conf'
student_threshold = 90

# Get some directory information
current_dir_name = os.path.basename(os.getcwd())  # Get the dir name to check if it matches 'Projets-QCM'
current_full_path = os.path.dirname(os.getcwd()) + '/' + current_dir_name  # Get the full path in case it matches
today = datetime.datetime.now().strftime('%d/%m/%Y')
report_author = 'Gregory Furter'
colour_palette = {'heading_1': (23, 55, 83, 255), 'heading_2': (109, 174, 219, 55), 'heading_3': (40, 146, 215, 55)}


def get_definitions():
    """
    Get the definitions from the definitions.json file
    :return: a dictionary of definitions
    """
    file_path = "definitions.json"

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


def get_settings(filename):
    """
    Read the settings from a configuration file, or create a new one if it doesn't exist.

    :param filename: the name of the configuration file:
    :return: values of a dictionary of the settings
    """

    if not os.path.isfile(filename):
        create_file = input(f"The file {filename} doesn't exist. Do you want to create it? (y/n): ")
        default_dir = '/path/to/project/dir'
        if create_file.lower() == 'y':
            look_for_path = input('Do you want to automatically search for the "Projets-QCM" directory? (y/n)')
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
                'openai.api_key': ''
            }
            with open(filename, 'w') as configfile:
                config.write(configfile)
        else:
            raise ValueError(f"The file {filename} was not created.")
    else:
        config = configparser.ConfigParser()
        config.read(filename)
        if not openai:
            settings = {
                'projects_dir': config.get('DEFAULT', 'projects_dir'),
                'openai.api_key': config.get('DEFAULT', 'openai.api_key')
            }
        else:
            settings = {
                'projects_dir': config.get('DEFAULT', 'projects_dir'),
                'openai.api_key': openai.api_key
            }
        if not settings['openai.api_key']:
            api_key = input("Please enter your OpenAI API key: ")
            config.set('DEFAULT', 'openai.api_key', api_key)
            with open(filename, 'w') as configfile:
                config.write(configfile)
            settings['openai.api_key'] = api_key
        # print("Settings returned: ")
        # print(settings)
        return settings.values()


def get_project_directories(path):
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
        exit(1)
    else:
        subdirectories = next(os.walk(path))[1]
        subdirectories.remove('_Archive')
        subdirectories.sort()

        while True:
            # display numbered list of subdirectories
            print("Here's a list of current projects:")
            for i, subdirectory in enumerate(subdirectories):
                print(f"{i + 1}. {subdirectory}")

            # prompt user to select a subdirectory
            selection = input("Enter the number of the project you'd like to select: ")

            # validate user input
            while not selection.isdigit() or int(selection) not in range(0, len(subdirectories) + 1):
                selection = input(
                    "Invalid input. Enter the number of the project you'd like to select (type 0 for list): ")

            # If user input is 0, then print the list again
            if selection == '0':
                continue

            # store the path to the selected project
            selected_path = os.path.join(path, subdirectories[int(selection) - 1])
            return selected_path


def get_questions_url(path):
    """
    Get the URL of the questions-tex file by parsing the source.tex file
    :param path:
    :return: URL to the questions.tex file as a string
    """

    # Open the file and read its content
    with open(path, 'r') as f:
        content = f.read()

    # Use regular expressions to search for the values you want
    year = re.search(r'\\def\\ExYear\{(\d+)\}', content).group(1)
    fold = re.search(r'\\def\\ExFold\{(.+?)\}', content).group(1)
    prof_fold = re.search(r'\\def\\ProfFold\{(.+?)\}', content).group(1)

    # Print the values to verify they were captured correctly
    question_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(path)))) + '/' + prof_fold + '/' + year + '/' + fold + '/'

    return question_dir


def get_tables(db):
    """
    Get the scoring data by querying 'scoring.sqlite' from the project directory:

    * pd_mark contains the marks obtained by the students:
        - student: student id in the database
        - total: total points obtained by the student
        - max: maximum obtainable mark
        - mark: mark obtained by the student
    * pd_score are the scores obtained by each student for each question and the status of the question:
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
    * pd_question is the list of all individual questions with additional data (some columns optional):
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
    pd_score = pd.read_sql_query(f"""SELECT ss.student, ss.question, st.title, ss.score, ss.why, ss.max
                                FROM scoring_score ss
                                JOIN scoring_title st ON ss.question = st.question
                                WHERE ss.question > {student_code_length}""", conn)
    pd_answer = pd.read_sql_query(f"""SELECT DISTINCT question, answer, correct, strategy
                                  FROM scoring_answer WHERE question > {student_code_length}""", conn)
    pd_variables = pd.read_sql_query("SELECT * FROM scoring_variables", conn, index_col='name')

    # close the database connection
    conn.close()

    if pd_mark.empty:
        print("Error: No mark has been recorded in the database")
        exit()

    # print(f"pd_question1: \n{pd_question.head()}")
    # Clean the scores to keep track of Cancelled (C), Floored (P), Empty (V) and Error (E) questions
    why = pd.get_dummies(pd_score['why'])
    pd_score = pd.concat([pd_score, why], axis=1)
    pd_score.drop('why', axis=1, inplace=True)
    pd_score.rename(columns={'': 'replied', 'C': 'cancelled', 'P': 'floored', 'V': 'empty', 'E': 'error'}, inplace=True)
    pd_score['correct'] = pd_score.apply(lambda row: 1 if row['score'] == row['max'] else 0, axis=1)

    # Create pd_question as a pivot table of pd_scores. It now contains the following columns:
    # question title  cancelled  correct  empty error floored  max  replied  score (some columns are optional)
    pd_question = pd_score.pivot_table(index=['question', 'title'],
                                       values=pd_score.columns[3:],
                                       aggfunc='sum',
                                       fill_value=0).reset_index()

    # Get the list of columns to calculate the number of times a question has been presented.
    cols_for_pres = [col for col in ['cancelled', 'empty', 'replied', 'error'] if col in pd_question.columns]
    pd_question['presented'] = pd_question[cols_for_pres].sum(axis=1)
    # Get the list of columns for calculate the number of times a question has been replied or left empty...
    cols_for_diff = [col for col in ['floored', 'empty', 'replied', 'error'] if col in pd_question.columns]
    # Calculate the difficulty of each question
    pd_question['difficulty'] = pd_question['correct'] / pd_question[cols_for_diff].sum(axis=1)
    # Now the columns are: ['question', 'title', 'cancelled', 'correct', 'empty', 'error', 'max', 'replied', 'score',
    # 'presented', 'difficulty'] - some columns are optional

    # Apply specific operations to pd_answer before returning it
    pd_answer['correct'] = pd_answer.apply(lambda x: 1 if (x['correct'] == 1) or ('1' in x['strategy']) else 0, axis=1)

    return pd_mark, pd_score, pd_variables, pd_question, pd_answer


def ticked(row):
    """
    Define if an answer box has been ticked by looking at the darkness of the box compared to the threshold.
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
    FRom this dataframe, we can determine if an answer box has been ticked based on the darkness of the box \
    compared to the threshold.
    * pd_capture contains all the questions and answers with indication of correct answers:
        - question: question id in the database
        - answer: answer number for the questions (1=A, 2=B, ...)
        - correct: 1 if the answer is correct, 0 otherwise

    :param db: path to the database
    :return capture_table and questions and answers summary:
    """
    # create a connection to the database
    conn = sqlite3.connect(db)

    pd_capture = pd.read_sql_query(f"""SELECT student, id_a AS 'question', id_b AS 'answer', total, black, manual 
                                    FROM capture_zone 
                                    WHERE type = 4 AND id_a > {student_code_length}""", conn)

    # close the database connection
    conn.close()

    # Apply specific operations to dataframes before returning them
    # pd_capture
    pd_capture['ticked'] = pd_capture.apply(ticked, axis=1)

    # pd_items
    pd_items = pd_capture.groupby(['question', 'answer'])['ticked'].sum().reset_index().sort_values(
        by=['question', 'answer'])
    pd_items['correct'] = pd_items.apply(lambda row: answer_df.loc[
        (answer_df['question'] == row['question']) & (answer_df['answer'] == row['answer']), 'correct'].values[0],
                                         axis=1)
    pd_items = pd_items.merge(question_df[['question', 'title']], left_on='question', right_on='question')
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
    n = mark_df['student'].nunique()
    number_of_questions = question_df['title'].nunique()
    max_possible_score = float(variables_df['value']['mark_max'])
    min_achieved_score = mark_df['mark'].min()
    max_achieved_score = mark_df['mark'].max()
    mean_score = mark_df['mark'].mean()
    median_score = mark_df['mark'].median()
    mode_score = mark_df['mark'].mode().iloc[0]
    std_score = mark_df['mark'].std()
    var_score = mark_df['mark'].var()
    sem_score = stats.sem(mark_df['mark'])
    sem_measurement = std_score / (n ** 0.5)
    skewness = mark_df['mark'].skew()
    kurtosis = mark_df['mark'].kurtosis()
    alpha = pg.cronbach_alpha(data=(score_df.select_dtypes(include=['int64', 'float64'])
                                    if 'cancelled' not in score_df.columns
                                    else score_df[score_df['cancelled'] == 0].select_dtypes(
        include=['int64', 'float64'])),
                              items='question',
                              scores='score')

    # create a dictionary to store the statistics
    stats_dict = {
        'Number of examinees': n,
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
        'Test reliability (Cronbach\'s Alpha)': alpha

    }

    # create a Pandas DataFrame from the dictionary
    pd_stats = pd.DataFrame.from_dict(stats_dict, orient='index', columns=['Value'])

    return pd_stats


def read_exam_file(filename):
    """
    This one is supposed to read the latex file containing the questions and answers, extract the questions and
    answers and store them in a dictionary.
    This should be used to match the questions and answers to the questions and results in the database.
    The goal is to be able to display not only the questions identifiers, but the complete wording of questions
    and answers for better and faster reading of the report.
    This does not work yet but has to be worked on at a later date.
    :param filename: path to the latex file with questions and answers.
    :return:
    """
    with open(filename, 'r') as f:
        exam_str = f.read()

    # Use regex to find all the elements and their content
    pattern = re.compile(r'\\element\{(.*?)\}\{(.*?)\\end\{question(mult)?\}\}', re.DOTALL)
    matches = pattern.findall(exam_str)

    print(matches[:5])
    # Create a dictionary to store the groups and their questions
    groups = {}

    # Loop through the matches and extract the group name and questions
    for group_name, group_questions in matches:
        # Use regex to extract each individual question and its parts
        question_pattern = re.compile(r'\\begin{question(.*?)\\end{question}', re.DOTALL)
        question_matches = question_pattern.findall(group_questions)

        # Loop through the question matches and extract the question text and choices
        questions = []
        for question_match in question_matches:
            # Extract the question text
            text_pattern = re.compile(r'\\begin{question}(.*?)\\begin{choices}', re.DOTALL)
            text_match = text_pattern.search(question_match)
            question_text = text_match.group(1).strip()

            # Extract the choices and their correctness
            choices_pattern = re.compile(r'\\(correct|wrong)choice{(.*?)}', re.DOTALL)
            choices_matches = choices_pattern.findall(question_match)
            choices = []
            for is_correct, choice_text in choices_matches:
                choices.append({'text': choice_text.strip(), 'is_correct': is_correct == 'correct'})

            # Add the question and its parts to the list of questions
            questions.append({'text': question_text, 'choices': choices})

        # Add the list of questions to the appropriate group
        groups[group_name] = questions

    return groups


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
                                on=['student'])

    # Merge questions scores and students mark, top quantile
    top_merged_df = pd.merge(top_27_df,
                             (score_df.select_dtypes(include=['int64', 'float64'])
                              if 'cancelled' not in score_df.columns
                              else score_df[score_df['cancelled'] == 0].
                              select_dtypes(include=['int64', 'float64'])),
                             on=['student'])

    # Group by question and answer, and calculate the mean mark for each group
    top_mean_df = top_merged_df.groupby(['question', 'student']).mean()
    bottom_mean_df = bottom_merged_df.groupby(['question', 'student']).mean()

    # Calculate the discrimination index for each question
    discrimination = []  # Create a list to store the results
    nb_in_groups = round(len(mark_df) * 0.27)
    for question in top_mean_df.index.levels[0]:
        # print(question)
        discr_index = (len(top_mean_df.loc[question][top_mean_df.loc[question]['score'] == 1]) - len(
            bottom_mean_df.loc[question][bottom_mean_df.loc[question]['score'] == 1])) / nb_in_groups
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
    top_sum_df = top_merged_df[['question', 'answer', 'ticked']].groupby(['question', 'answer']).sum()
    bottom_sum_df = bottom_merged_df[['question', 'answer', 'ticked']].groupby(['question', 'answer']).sum()

    # Calculate the discrimination index for each question
    discrimination = {'question': [], 'answer': [], 'discrimination': []}  # Create a dictionary to store the results
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
        merged_df = pd.merge(score_df[score_df['cancelled'] == False], mark_df, on='student')
    else:
        merged_df = pd.merge(score_df, mark_df, on='student')
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
        merged_df = pd.merge(capture_df, score_df[['student', 'question', 'cancelled']], on=['student', 'question'])
        merged_df = merged_df[merged_df['cancelled'] == False].merge(mark_df[['student', 'mark']], on='student')
    else:
        merged_df = capture_df.merge(mark_df[['student', 'mark']], on='student')
    outcome_corr = {'question': [], 'answer': [], 'correlation': []}
    questions = merged_df['question'].unique()
    for question in questions:
        answers = merged_df['answer'][merged_df['question'] == question].unique()
        for answer in answers:
            item_scores = merged_df.loc[(merged_df['question'] == question) & (merged_df['answer'] == answer), 'ticked']
            total_scores = merged_df.loc[(merged_df['question'] == question) & (merged_df['answer'] == answer), 'mark']
            correlation = stats.pointbiserialr(item_scores.values, total_scores.values)
            outcome_corr['question'].append(question)
            outcome_corr['answer'].append(answer)
            outcome_corr['correlation'].append(correlation[0])
    outcome_corr_df = pd.DataFrame.from_dict(outcome_corr)
    return outcome_corr_df


def plot_difficulty_and_discrimination():
    """
    Plot the difficulty and discrimination index for each question.
    This should be be streamlined and done differently at a later stage:

    - Do one plot at a time, not 4 at once.
    - Save each individual plot as a standalone file to be integrated in the PDF report.
    :return:
    """
    # Create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Calculate the number of bins based on the maximum and minimum marks
    plot_bins = int(float(stats_df.loc['Maximum achieved mark', 'Value'])
                    - float(stats_df.loc['Minimum achieved mark', 'Value']))
    # create a histogram of the 'mark' column
    sns.histplot(mark_df['mark'], kde=True, ax=axs[0, 0], bins=plot_bins)
    axs[0, 0].set_title('Frequency of Marks')
    # create a histogram of difficulty levels
    sns.histplot(question_df['difficulty'], bins=30, ax=axs[1, 0], color='blue')
    axs[1, 0].set_xlabel('Difficulty level\n(higher is easier)')
    axs[1, 0].set_ylabel('Number of questions')
    axs[1, 0].set_title('Difficulty Levels')

    # create a histogram of discrimination indices
    sns.histplot(question_df['discrimination'], bins=30, ax=axs[0, 1], color='orange')
    axs[0, 1].set_xlabel('Discrimination index\n(the higher the better)')
    axs[0, 1].set_ylabel('Number of questions')
    axs[0, 1].set_title('Discrimination Indices')

    # Set the color of the bars in the first histogram
    for patch in axs[1, 0].patches[:13]:
        patch.set_color('tab:red')
    for patch in axs[1, 0].patches[13:23]:
        patch.set_color('tab:blue')
    for patch in axs[1, 0].patches[23:]:
        patch.set_color('tab:green')

    # Set the color of the bars in the second histogram
    for patch in axs[0, 1].patches[:13]:
        patch.set_color('tab:orange')
    for patch in axs[0, 1].patches[13:23]:
        patch.set_color('tab:blue')
    for patch in axs[0, 1].patches[23:]:
        patch.set_color('tab:green')

    # Plot the questions on a scatter plot
    sns.scatterplot(y='difficulty', x='discrimination', data=question_df, ax=axs[1, 1], color='purple')
    sns.jointplot(y='difficulty', x='discrimination', data=question_df, color='purple', kind='reg')

    # Set the axis labels
    axs[1, 1].set_ylabel('Difficulty (high is easy)')
    axs[1, 1].set_xlabel('Discrimination (the higher the better)')
    axs[1, 1].set_title('Difficulty vs Discrimination')

    # Set the axis limits for discrimination plot
    # axs[1, 1].set_xlim(0, 1)
    axs[1, 1].set_ylim(0, 1)

    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    # save each subplot as an image file (does not work, saves the whole thing at once
    axs[0, 0].get_figure().savefig('marks.png')
    axs[0, 1].get_figure().savefig('discrimination.png')
    axs[1, 0].get_figure().savefig('difficulty.png')
    axs[1, 1].get_figure().savefig('correlation.png')

    # Show the plot
    plt.show()


def get_blob():
    """
    Generate a first level of analysis on the performance of the questions. This text can either be used as is in the
    report or passed to ChatGPT for a better wording.
    :return: a string of text describing the data and how to improve the exam questions.
    """
    intro = "According to the data collected, the following questions should probably be reviewed:\n"
    blb = ''
    if ('cancelled' in question_df.columns) \
            and (question_df[question_df['cancelled'] > question_df['presented'] / 1.2]['title'].values.size > 0):
        top_cancelled = question_df[question_df['cancelled']
                                    > question_df['presented'] / 1.2].sort_values('title')['title'].values
        if len(top_cancelled) > 1:
            blb += f"""- Questions  {', '.join(top_cancelled[:-1]) + ' and ' 
                                + top_cancelled[-1]} have been cancelled more than 80% of the time.\n"""
        else:
            blb += f"""- Question {top_cancelled[0]} has been cancelled more than 80% of the time.\n"""
    if ('empty' in question_df.columns) \
            and (question_df[question_df['empty'] > question_df['presented'] / 1.2].count()['title'] > 0):
        top_empty = question_df[question_df['empty']
                                > question_df['presented'] / 1.2].sort_values('title')['title'].values
        if len(top_empty) > 1:
            blb += f"""- Questions {', '.join(top_empty[:-1]) + ' and ' 
                                + top_empty[-1]} have been empty more than 80% of the time.\n"""
        else:
            blb += f"""- Question {top_empty[0]} has been empty more than 80% of the time.\n"""
    if ('discrimination' in question_df.columns) \
            and (question_df[question_df['discrimination'] < 0]['title'].values.size > 0):
        negative_discrimination = question_df[question_df['discrimination'] < 0].sort_values('title')['title'].values
        if len(negative_discrimination) > 1:
            blb += f"""- Questions {', '.join(negative_discrimination[:-1]) + ' and ' 
                        + negative_discrimination[-1]} have a negative discrimination.\n"""
        else:
            blb += f"- Question {negative_discrimination[0]} has a negative discrimination.\n"
    if items_df[items_df['ticked'] == 0]['title'].values.size > 0:
        not_ticked = items_df[items_df['ticked'] == 0]['title'].unique()
        not_ticked.sort()
        if len(not_ticked) > 1:
            blb += f"""- Questions {', '.join(not_ticked[:-1]) + ' and ' 
                        + not_ticked[-1]} have distractors that have never been chosen.\n"""
        else:
            blb += f"- Question {not_ticked[0]} has distractors that have never been chosen.\n"
    if len(blb) > 0:
        blb = intro + blb
    else:
        blb = "According to the data collected, there are no questions to review based on their performance."
    return blb



def get_gpt_text(prompt):
    response = openai.Completion.create(engine='text-davinci-003',
                                        prompt=prompt,
                                        max_tokens=500,
                                        temperature=0.5)
    return response['choices'][0]['text']


def read_latex(questionfile):
    import re

    # Read the LaTeX file
    with open(questionfile, 'r') as f:
        latex_str = f.read()

    # Define regular expression patterns to match the question elements
    element_pattern = re.compile(r'\\element{(?P<group>\w+)}{(?P<latex>.+?)}', re.DOTALL)
    question_pattern = re.compile(
        r'\\begin{question(?P<mult>mult)?}{(?P<id>\w+)}\n\s*(?P<text>.+?)\n\s*\\end{question(?P=mult)}', re.DOTALL)
    choice_pattern = re.compile(r'\\(?:wrong|correct)choice{(.+?)}')

    # Parse the LaTeX code and extract the questions and choices
    questions = []
    for match in element_pattern.finditer(latex_str):
        group = match.group('group')
        latex = match.group('latex')
        for question_match in question_pattern.finditer(latex):
            q_id = question_match.group('id')
            q_text = question_match.group('text')
            q_text = LatexNodes2Text().latex_to_text(q_text)  # Convert LaTeX to plain text
            q_text = q_text.strip()  # Remove leading/trailing whitespace
            q_is_mult = question_match.group('mult') is not None
            q_choices = []
            for choice_match in choice_pattern.finditer(q_text):
                choice_text = choice_match.group(1)
                choice_text = LatexNodes2Text().latex_to_text(choice_text)  # Convert LaTeX to plain text
                choice_text = choice_text.strip()  # Remove leading/trailing whitespace
                is_correct = 'correct' in choice_match.group(0)
                q_choices.append((choice_text, is_correct))
            questions.append((group, q_id, q_text, q_is_mult, q_choices))
    return questions


# Press the green button in the gutter to run the script.
def get_student_code_length(db):
    """
    Get the number of student code boxes
    :param db: path to the database
    :return: Length of the student code as an integer
    """
    # create a connection to the database
    conn = sqlite3.connect(db)

    # get the number boxes for the student code so we can only query the real questions
    scl = pd.read_sql_query("SELECT COUNT(*) FROM scoring_title WHERE title LIKE '%student.number%';"
                            , conn).iloc[0][0]
    # close the database connection
    conn.close()
    return scl


if __name__ == '__main__':

    directory_path, openai.api_key = get_settings(config_filename)
    # Get the Project directory and questions file paths
    amcProject = get_project_directories(directory_path)
    scoring_path = amcProject + '/data/scoring.sqlite'
    capture_path = amcProject + '/data/capture.sqlite'
    amcProject_name = glob.glob(amcProject, recursive=False)[0].split('/')[-1]
    source_file_path = glob.glob(amcProject + '/*source*.tex', recursive=True)
    # question_path = glob.glob(get_questions_url(source_file_path[0]) + '/*questions*.tex', recursive=False)[0]

    question_data_columns = ['presented', 'cancelled', 'replied', 'correct', 'empty', 'error', ]
    question_analysis_columns = ['difficulty', 'discrimination', 'correlation', ]
    outcome_data_columns = ['answer', 'correct', 'ticked', 'discrimination', ]

    definitions = get_definitions()
    # Issue an error and terminate if the scoring database does not exist
    if not os.path.exists(scoring_path):
        print("Error: the database does not exist!")
        sys.exit(1)  # terminate the program with an error code

    student_code_length = get_student_code_length(scoring_path)

    # get the data from the databases
    mark_df, score_df, variables_df, question_df, answer_df = get_tables(scoring_path)
    capture_df, items_df = get_capture_table(capture_path)
    # display general statistics about the exam
    stats_df = general_stats()

    print('\nGeneral Statistics')
    print(stats_df)
    # Get item and outcome discrimination if the number of examinees is greater than 99
    if stats_df.loc['Number of examinees'][0] > student_threshold:
        # Create two student dataframes based on the quantile values. They should have the same number of students
        # This should probably be done in a smarter way, outside, in order to be used for item discrimination.
        top_27_df = mark_df.sort_values(by=['mark'], ascending=False).head(round(len(mark_df) * 0.27))
        bottom_27_df = mark_df.sort_values(by=['mark'], ascending=False).tail(round(len(mark_df) * 0.27))

        question_df['discrimination'] = questions_discrimination()
        items_discr = items_discrimination()
        items_df = items_df.merge(items_discr[['question', 'answer', 'discrimination']], on=['question', 'answer'])

        # plot_difficulty_and_discrimination()

        question_corr = question_df[['difficulty', 'discrimination']].corr()
        print(f'\nCorrelation between difficulty and discrimination:\n{question_corr}')

    # Get item (question) correlation
    item_correlation = get_item_correlation()
    question_df['correlation'] = question_df['title'].apply(lambda row: item_correlation.loc[row]['correlation'])

    # Get outcome (answer) correlation
    outcome_correlation = get_outcome_correlation()
    items_df = items_df.merge(outcome_correlation, on=['question', 'answer'])

    print(f"\nList of questions (question_df):\n{question_df.head()}")
    print(f"\nList of answers (answer_df):\n{answer_df.head()}")
    print(f"\nList of items (items_df):\n{items_df.sort_values(by='title').head()}")
    print(f"\nList of variables (variables_df):\n{variables_df}")
    print(f"\nList of capture (capture_df):\n{capture_df.head()}")
    print(f"\nList of mark (mark_df):\n{mark_df.head()}")
    print(f"\nList of score (score_df):\n{score_df.head()}")
    # print(f"\nitems discrimination: \n{items_discrimination()}")

    blob = get_blob()
    report_params = {
        'project_name': amcProject_name,
        'questions': question_df,
        'items': items_df,
        'author': report_author,
        'stats': stats_df,
        'threshold': student_threshold,
        'marks': mark_df,
        'definitions': definitions,
        'palette': colour_palette,
        'blob': blob,
    }

    plot_charts(report_params)

    generate_pdf_report(report_params)

    exit(0)

    # Saving dataframes to disk to be used in tests
    question_df.to_pickle('./question_df.pkl')
    answer_df.to_pickle('./answer_df.pkl')
    items_df.to_pickle('./items_df.pkl')
    variables_df.to_pickle('./variables_df.pkl')
    capture_df.to_pickle('./capture_df.pkl')
    mark_df.to_pickle('./mark_df.pkl')
    score_df.to_pickle('./score_df.pkl')
    stats_df.to_pickle('./stats_df.pkl')

    # print(f"list of top performing students:\n{top_27_df}")
    # marks_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), mark_df, verbose=True)
    # scores_agent = create_pandas_dataframe_agent(OpenAI(temperature=0.2), question_df, verbose=True)

    # marks_agent.run("Give relevant statistic figures for these exam grades.")
    # scores_agent.run("Give relevant statistical figures for these exam questions. Return the 5 best performing and 5 "
    #                 "least performing questions according to the Classical Test Theory.")

    # agent = create_pandas_dataframe_agent(OpenAI(temperature=0), question_df.sort_values(by='title'), verbose=True)

    # agent.run("Give a statistical explanation of these exam questions based on the difficulty level (higher is "
    #          "easier) and the discrimination index (higher is better). highlight the 5 best questions and the 5 "
    #          "worst questions")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

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

import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy import stats
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

config_filename = 'settings.conf'

# Get some directory information
current_dir_name = os.path.basename(os.getcwd())  # Get the dir name to check if it matches 'Projets-QCM'
current_full_path = os.path.dirname(os.getcwd()) + '/' + current_dir_name  # Get the full path in case it matches


def get_settings(filename):
    """
    Read the settings from a configuration file, or create a new one if it doesn't exist.

    :param filename: the name of the configuration file
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
        print("Settings returned: ")
        print(settings)
        return settings.values()


def get_project_directories(path):
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
            # print(f"The path to the selected project is: {selected_path}")
            break

    # returns the path to the
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
    list all the tables and columns of the file 'scoring.sqlite'
    from the project directory
    """
    # create a connection to the database
    conn = sqlite3.connect(db)

    # create a cursor object to execute SQL commands
    # cursor = conn.cursor()

    pd_mark = pd.read_sql_query("SELECT student, total, max, mark FROM scoring_mark", conn)
    pd_score = pd.read_sql_query("SELECT student, question, score, why, max "
                                 "FROM scoring_score WHERE question > 8", conn)
    pd_question = pd.read_sql_query("SELECT * FROM scoring_title WHERE question > 8", conn)
    pd_answer = pd.read_sql_query("SELECT DISTINCT question, answer, correct, strategy "
                                  "FROM scoring_answer WHERE question > 8", conn)
    pd_variables = pd.read_sql_query("SELECT * FROM scoring_variables", conn, index_col='name')
    pd_why = pd.read_sql_query("SELECT why, count(question) "
                               "FROM scoring_score WHERE question > 8 GROUP BY why", conn)

    if pd_mark.empty:
        print("Error: No mark has been recorded in the database")
        exit()

    # close the database connection
    conn.close()

    # Clean the scores to remove cancelled questions
    pd_score = pd_score[pd_score['why'] != 'C']
    pd_score = pd_score.drop('why', axis=1)

    # Apply specific operations to pd_answer before returning it
    pd_answer['correct'] = pd_answer.apply(lambda x: 1 if (x['correct'] == 1) or ('1' in x['strategy']) else 0, axis=1)

    return pd_mark, pd_score, pd_variables, pd_question, pd_answer, pd_why


def ticked(row):
    """
    Define if an answer box has been ticked by looking at the darknes of the box compared to the threshold.
    :param row:
    :return 1 or 0:
    """
    # Get thresholds to calculate ticked answers and get the item analysis
    darkness_bottom = float(variables_df.loc['darkness_threshold']['value'])
    darkness_top = float(variables_df.loc['darkness_threshold_up']['value'])

    # If the box has been manually ticked => correct = 1
    if row['manual'] == 1:
        return 1
    # If the box darkness is within the threshold => correct = 1
    elif row['total'] * darkness_bottom < row['black'] <= row['total'] * darkness_top:
        return 1
    else:
        return 0


def get_capture_table(db):
    '''
    list all the tables and columns of the file 'capture.sqlite'
    from the project directory
    :return capture_table and questions and answers summary:
    '''
    # create a connection to the database
    conn = sqlite3.connect(db)

    # create a cursor object to execute SQL commands
    cursor = conn.cursor()

    pd_capture = pd.read_sql_query("""SELECT student, id_a AS 'question', id_b AS 'answer', total, black, manual 
                                    FROM capture_zone 
                                    WHERE type = 4 AND id_a > 8""", conn)

    # close the database connection
    conn.close()

    # Apply specific operations to dataframes before returning them
    ## pd_capture
    pd_capture['ticked'] = pd_capture.apply(ticked, axis=1)

    ## pd_items
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
    alpha = pg.cronbach_alpha(data=score_df)

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
    :return: a list of discrtimination indices to be added as a column to question_df
    """
    # Create two student dataframes based on the quantile values. They should have the same number of students
    top_27_df = mark_df.sort_values(by=['mark'], ascending=False).head(round(len(mark_df) * 0.27))
    bottom_27_df = mark_df.sort_values(by=['mark'], ascending=False).tail(round(len(mark_df) * 0.27))

    print(top_27_df)
    # Merge questions scores and students mark, bottom quantile
    bottom_merged_df = pd.merge(bottom_27_df, score_df, on=['student'])
    bottom_merged_df.rename(columns={'max_x': 'max_points', 'max_y': 'max_score'}, inplace=True)

    # Merge questions scores and students mark, top quantile
    top_merged_df = pd.merge(top_27_df, score_df, on=['student'])
    top_merged_df.rename(columns={'max_x': 'max_points', 'max_y': 'max_score'}, inplace=True)
    print(f"\nTop merged df: \n{top_merged_df}")
    # Group by question and answer, and calculate the mean mark for each group
    top_mean_df = top_merged_df.groupby(['question', 'student']).mean()
    bottom_mean_df = bottom_merged_df.groupby(['question', 'student']).mean()
    print(f"\nTop mean df: \n{top_mean_df}")

    # Calculate the discrimination index for each question
    discrimination = []  # Create a list to store the results
    for question in top_mean_df.index.levels[0]:
        # print(question)
        discr_index = (len(top_mean_df.loc[question][top_mean_df.loc[question]['score'] == 1]) - len(
            bottom_mean_df.loc[question][bottom_mean_df.loc[question]['score'] == 1])) / round(len(mark_df) * 0.27)
        discrimination.append(discr_index)  # Add the result to the list

    # Add the discrimination indices to the question_df dataframe
    return discrimination


def difficulty_level():
    """
    Calculate the difficulty level for each question.
    Add a column 'difficulty' to the dataframe 'question_df' with the index for each question
    :return: nothing returned, question_df is modified This may need adjustment
    """
    merged_df = pd.merge(mark_df, score_df, on=['student'])
    difficulty = []
    # loop through each question and calculate the difficulty level
    for q in question_df['question']:
        # select the scores for the current question from the merged dataframe
        scores = merged_df.loc[merged_df['question'] == q, 'score']
        # Calculate the difficulty and add the result to the array
        difficulty.append(len(scores[scores > 0]) / len(scores))

    return difficulty


def plot_difficulty_and_discrimination():
    # Create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Calculate the number of bins based on the maximum and minimum marks
    plot_bins = int(float(stats_df.loc['Maximum achieved mark', 'Value']) - float(stats_df.loc['Minimum achieved mark', 'Value']))
    # create a histogram of the 'mark' column
    sns.histplot(mark_df['mark'], kde=True,ax=axs[0, 0], bins=plot_bins)
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
    #axs[1, 1].set_xlim(0, 1)
    axs[1, 1].set_ylim(0, 1)

    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    # save each subplot as an image file (does not work, saves the whole thing at once
    axs[0, 0].get_figure().savefig('marks.png')
    axs[0, 1].get_figure().savefig('discrimination.png')
    axs[1, 0].get_figure().savefig('difficulty.png')
    axs[1, 1].get_figure().savefig('correlation.png')

    # Show the plot
    plt.show()


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
if __name__ == '__main__':

    directory_path, openai.api_key = get_settings(config_filename)
    # Get the Project directory and questions file paths
    amcProject = get_project_directories(directory_path)
    scoring_path = amcProject + '/data/scoring.sqlite'
    capture_path = amcProject + '/data/capture.sqlite'
    source_file_path = glob.glob(amcProject + '/*source.tex', recursive=True)
    question_path = glob.glob(get_questions_url(source_file_path[0]) + '/*questions.tex', recursive=False)[0]

    # Issue an error and terminate if the scoring database does not exist
    if not os.path.exists(scoring_path):
        print("Error: the database does not exist!")
        sys.exit(1)  # terminate the program with an error code

    # get the data from the databases
    mark_df, score_df, variables_df, question_df, answer_df, why_df = get_tables(scoring_path)
    capture_df, items_df = get_capture_table(capture_path)
    # display general statistics about the exam
    stats_df = general_stats()

    print('\nGeneral Statistics')
    print(stats_df)

    question_df['discrimination'] = questions_discrimination()

    question_df['difficulty'] = difficulty_level()

    print(f'\n{why_df}\n')

    plot_difficulty_and_discrimination()

    print(f"\nList of questions:\n{question_df.sort_values(by='title')}")

    question_corr = question_df[['difficulty', 'discrimination']].corr()
    print(f'\nCorrelation between difficulty and discrimination:\n{question_corr}')


    #print(f"list of top performing students:\n{top_27_df}")
    #marks_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), mark_df, verbose=True)
    # scores_agent = create_pandas_dataframe_agent(OpenAI(temperature=0.2), question_df, verbose=True)

    #marks_agent.run("Give relevant statistic figures for these exam grades.")
    #scores_agent.run("Give relevant statistical figures for these exam questions. Return the 5 best performing and 5 "
    #                 "least performing questions according to the Classical Test Theory.")

    # agent = create_pandas_dataframe_agent(OpenAI(temperature=0), question_df.sort_values(by='title'), verbose=True)

    # agent.run("Give a statistical explanation of these exam questions based on the difficulty level (higher is "
    #          "easier) and the discrimination index (higher is better). highlight the 5 best questions and the 5 "
    #          "worst questions")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



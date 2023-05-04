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

from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode
from pylatexenc.latex2text import LatexNodes2Text

# Try to get API KEY from ENV
openai.api_key = os.getenv('OPENAI_API_KEY')

sns.set_theme()
sns.set_style('darkgrid')
sns.set_style()
sns.color_palette("tab10")

config_filename = 'settings.conf'

# specify the directory to browse
# directory_path = "/Users/greg/Dropbox/01-QCM/_AMC/Projets-QCM"
home_dir = os.getenv('HOME')  # Get HOME dir to look for 'Projets-QCM'
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
        print(settings)
        return settings.values()

def get_project_directories(path):
    # list sub-directories and sorts them
    subdirectories = next(os.walk(directory_path))[1]
    subdirectories.remove('_Archive')
    subdirectories.sort()

    while True:
        # display numbered list of sub-directories
        print("Here's a list of current projects:")
        for i, subdirectory in enumerate(subdirectories):
            print(f"{i + 1}. {subdirectory}")

        # prompt user to select a sub-directory
        selection = input("Enter the number of the sub-directory you'd like to select: ")

        # validate user input
        while not selection.isdigit() or int(selection) not in range(0, len(subdirectories) + 1):
            selection = input(
                "Invalid input. Enter the number of the sub-directory you'd like to select (type 0 for list): ")

        # If user input is 0, then print the list again
        if selection == '0':
            continue

        # store the path to the selected sub-directory
        selected_path = os.path.join(directory_path, subdirectories[int(selection) - 1])
        # print(f"The path to the selected sub-directory is: {selected_path}")
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

    pd_mark = pd.read_sql_query("SELECT * FROM scoring_mark", conn)
    pd_score = pd.read_sql_query("SELECT * FROM scoring_score WHERE question > 8", conn)
    pd_question = pd.read_sql_query("SELECT * FROM scoring_title WHERE question > 8", conn)
    pd_answer = pd.read_sql_query("SELECT * FROM scoring_answer WHERE question > 8", conn)
    max_score = pd.read_sql_query("SELECT * FROM scoring_variables WHERE name = 'mark_max'", conn)
    pd_why = pd.read_sql_query("SELECT why, count(question) FROM scoring_score WHERE question > 8 GROUP BY why", conn)

    if pd_mark.empty:
        print("Error: No mark has been recorded in the database")
        exit()

    # close the database connection
    conn.close()

    # Clean the scores to remove cancelled questions
    pd_score = pd_score[pd_score['why'] != 'C']
    pd_score = pd_score.drop('why', axis=1)

    return pd_mark, pd_score, max_score, pd_question, pd_answer, pd_why


def general_stats():
    # compute summary statistics for the 'total' column
    # mark_summary = mark_df['mark'].describe()

    # print the summary statistics
    # print("\nSummary statistics for 'scoring_mark' table:")
    # print(mark_summary)

    # compute the frequency of each 'type' value in the 'scoring_score' table
    # score_summary = score_df['score'].describe()

    # print the frequency counts
    # print("\nSummary statistics for 'scoring_score' table:")
    # print(score_summary)

    # compute the statistics
    n = mark_df['student'].nunique()
    max_possible_score = max_df['value'].max()
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
    alpha = pg.cronbach_alpha(data=mark_df)

    # create a dictionary to store the statistics
    stats_dict = {
        'Number of examinees': n,
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

    # create a histogram of the 'mark' column
    plot_bins = int(float(max_achieved_score) - float(min_achieved_score))
    # plt.hist(mark_df['mark'], bins=plot_bins)
    # plt.title('Histogram of Scores')
    # plt.xlabel('Mark')
    # plt.ylabel('Frequency')
    sns.histplot(mark_df['mark'], kde=True, bins=plot_bins)
    return pd_stats

import re

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

def discrimination_index():
    # Create two student dataframes based on the quantile values. They should have the same number of students
    top_27_df = mark_df.sort_values(by=['mark'], ascending=False).head(round(len(mark_df) * 0.27))
    bottom_27_df = mark_df.sort_values(by=['mark'], ascending=False).tail(round(len(mark_df) * 0.27))

    # Merge questions scores and students mark, bottom quantile
    bottom_merged_df = pd.merge(bottom_27_df, score_df, on=['student', 'copy'])

    # Merge questions scores and students mark, top quantile
    top_merged_df = pd.merge(top_27_df, score_df, on=['student', 'copy'])

    # Group by question and answer, and calculate the mean mark for each group
    top_mean_df = top_merged_df.groupby(['question', 'student']).mean()
    bottom_mean_df = bottom_merged_df.groupby(['question', 'student']).mean()

    # Calculate the discrimination index for each question
    discrimination = []  # Create a list to store the results
    for question in top_mean_df.index.levels[0]:
        # print(question)
        discrimination_index = (len(top_mean_df.loc[question][top_mean_df.loc[question]['score'] == 1]) - len(
            bottom_mean_df.loc[question][bottom_mean_df.loc[question]['score'] == 1])) / round(len(mark_df) * 0.27)
        discrimination.append(discrimination_index)  # Add the result to the list
        discrimination_index = round(discrimination_index, 2)
        # print(f"Discrimination index for question {question - 8}: {discrimination_index}")

    # Add the discrimination index to the question_df dataframe
    question_df['discrimination'] = discrimination


def difficulty_level():
    merged_df = pd.merge(mark_df, score_df, on=['student', 'copy'])
    difficulty = []
    # loop through each question and calculate the difficulty level
    for q in question_df['question']:
        # select the scores for the current question from the merged dataframe
        scores = merged_df.loc[merged_df['question'] == q, 'score']
        # Calculate the difficulty and add the result to the array
        difficulty.append(len(scores[scores > 0]) / len(scores))

    question_df['difficulty'] = difficulty


def plot_difficulty_and_discrimination():
    # Create figure with two subplots
    # # Create figure with two subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    #
    # # Plot histogram of difficulty levels
    # N, bins, patches = ax1.hist(question_df['difficulty'], bins=30)
    # ax1.set_xlabel('Difficulty level\n(higher is easier)')
    # ax1.set_ylabel('Number of questions')
    # ax1.set_title('Difficulty levels')
    #
    # for i in range(0, 13):
    #     patches[i].set_facecolor('r')
    # for i in range(13, 23):
    #     patches[i].set_facecolor('b')
    # for i in range(23, len(patches)):
    #     patches[i].set_facecolor('g')
    #
    # # Plot histogram of discrimination indices
    # ax2.hist(question_df['discrimination'], bins=30)
    # ax2.set_xlabel('Discrimination index\n(the higher the better)')
    # ax2.set_ylabel('Number of questions')
    # ax2.set_title('Discrimination indices')
    #
    # # Plot the questions on a scatter plot
    # fig, ax3 = plt.subplots()
    # ax3.scatter(question_df['difficulty'], question_df['discrimination'])
    #
    # # Set the axis labels
    # ax3.set_xlabel('Difficulty (high is easy)')
    # ax3.set_ylabel('Discrimination (the higher the better)')
    #
    # # Set the axis limits
    # ax3.set_xlim(0, 1)
    # ax3.set_ylim(-1, 1)
    #
    # # Show the plot
    # plt.show()
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot histogram of difficulty levels
    sns.histplot(question_df['difficulty'], bins=30, ax=ax1, color='blue')
    ax1.set_xlabel('Difficulty level\n(higher is easier)')
    ax1.set_ylabel('Number of questions')
    ax1.set_title('Difficulty levels')

    # Plot histogram of discrimination indices
    sns.histplot(question_df['discrimination'], bins=30, ax=ax2, color='orange')
    ax2.set_xlabel('Discrimination index\n(the higher the better)')
    ax2.set_ylabel('Number of questions')
    ax2.set_title('Discrimination indices')

    # Set the color of the bars in the first histogram
    for patch in ax1.patches[:13]:
        patch.set_color('tab:red')
    for patch in ax1.patches[13:23]:
        patch.set_color('tab:blue')
    for patch in ax1.patches[23:]:
        patch.set_color('tab:green')

    # Set the color of the bars in the second histogram
    for patch in ax2.patches[:13]:
        patch.set_color('tab:orange')
    for patch in ax2.patches[13:23]:
        patch.set_color('tab:blue')
    for patch in ax2.patches[23:]:
        patch.set_color('tab:green')

    # Plot the questions on a scatter plot
    fig, ax3 = plt.subplots()
    sns.scatterplot(x='difficulty', y='discrimination', data=question_df, ax=ax3, color='purple')

    # Set the axis labels
    ax3.set_xlabel('Difficulty (high is easy)')
    ax3.set_ylabel('Discrimination (the higher the better)')

    # Set the axis limits
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-1, 1)

    # Show the plot
    plt.show()


def get_gpt_text(prompt):
    response = openai.Completion.create(engine='text-davinci-003',
                                        prompt=prompt,
                                        max_tokens=256,
                                        temperature=0.7)
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
    dataPath = amcProject + '/data/scoring.sqlite'
    source_file_path = glob.glob(amcProject + '/*source.tex', recursive=True)
    question_path = glob.glob(get_questions_url(source_file_path[0]) + '/*questions.tex', recursive=False)[0]

    # Issue an error and terminate if the scoring database does not exist
    if not os.path.exists(dataPath):
        print("Error: the database does not exist!")
        sys.exit(1)  # terminate the program with an error code

    # get the data from the database
    mark_df, score_df, max_df, question_df, answer_df, why_df = get_tables(dataPath)

    # display general statistics about the exam
    stats_df = general_stats()

    print(stats_df)

    discrimination_index()

    difficulty_level()

    print(why_df)

    plot_difficulty_and_discrimination()

    print(f"\n{question_df.sort_values(by='title')}")
    print(source_file_path[0])
    print(question_path)

    questions = read_latex(question_path)

    print(questions)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#!/usr/bin/env python3
from configparser import ConfigParser
import glob
import os
import sqlite3
import sys
import json
import datetime
import shutil
import subprocess
import platform

import matplotlib
import matplotlib.pyplot as plt
from openai import OpenAI
import pandas as pd
import seaborn as sns
import numpy as np

from scipy import stats
from report import generate_pdf_report
from icecream import install, ic

matplotlib.use('agg')

DEBUG: int = 1  # Set to 1 for debugging, meaning not using OpenAI

CONFIG: str = 'settings.conf'
ASSISTANT: str = 'asst_a2p7Kfa3Q3fyQbpBX1gaMrPG'
TEMPERATURE: float = 0.2
STATS_PROMPT = """You are a Data Scientist, specialised in the Classical Test Theory. 
Give a short qualitative  explanation about the overall exam results. 
Don't go into technical details, focus on meaning.
Don't give definitions of the elements, just explain what they mean in the current context.
Don't mention the Classical Test Theory in your reply.
Don't introduce your answer."""

NBQ: str = 'Number of questions'


class LLM:

    def __init__(self, stats_table: pd.DataFrame, temp: float = 0.1, assistant_id: str = ASSISTANT) -> None:
        self.client = OpenAI()
        self.temp = temp
        self.assistant_id = assistant_id
        self.table: pd.DataFrame = stats_table
        self.assistant = self.client.beta.assistants.retrieve(self.assistant_id)
        self.thread = self._thread()
        self.response: str = self._response()

    def _thread(self):
        # Step 5: Create a thread to query
        """
        Creates a thread to query with the OpenAI assistant.

        The thread is created with an initial message from the user instructing the assistant
        to check the exam paper for incorrect answers. The message also specifies the format
        of the response and what to do with correct answers. The uploaded file is also attached
        to the message.

        Returns:
            thread: The created thread object.

        Raises:
            Exception: If there is an error creating the thread.
        """
        print("Creating thread...")
        try:
            thread = self.client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        'content': f"Summarise the following statistics so that they are easy to "
                                   f"understand. Round the numbers in your answer and give a summary at"
                                   f" the end:\n{self.table}"
                    }
                ]
            )
        except Exception as err:
            print(f"Cannot create thread: \n{err}")
            sys.exit()
        return thread

    def _response(self) -> str:
        """
        Create a run and get the assistant's response

        After the thread is created, use the `create_and_poll` method to create a run and wait for
        the assistant to respond. The content of the assistant's response is extracted and returned.

        Returns:
            str: The content of the assistant's response.

        Raises:
            Exception: If there is an error running the assistant.
        """
        print("Asking assistant...")
        try:
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                temperature=self.temp
            )
            ic(run.status)
            ic(run)
        except Exception as err:
            print(f"Assistant failed to run: \n {err}")
            print(f"Deleting thread id '{self.thread.id}'...")
            self._delete()
            sys.exit()

        msg = list(self.client.beta.threads.messages.list(thread_id=self.thread.id, run_id=run.id))
        return msg[0].content[0].text.value

    def _delete(self) -> None:
        """
        Prompts the user to delete a thread from the OpenAI servers.

        Returns
        -------
        None
        """
        self.client.beta.threads.delete(self.thread.id)
        print(f"Thread id '{self.thread.id}' deleted.")


class Settings:

    def __init__(self, config_file: str) -> None:
        self.config_file: str = config_file
        if not os.path.isfile(config_file):
            self._create_config_file()
        self.config: ConfigParser = ConfigParser()
        self.config.read(config_file)
        self.projects: str = os.path.expanduser("~") + self.config.get('DEFAULT', 'projects_dir')
        self.threshold: int = int(self.config.get('DEFAULT', 'student_threshold'))
        self.company_name = self.config.get('DEFAULT', 'company_name')
        self.company_url = self.config.get('DEFAULT', 'company_url')

    def _create_config_file(self):
        create_file = input(f"The file {self.config_file} doesn't exist. Do you want to create it? (Y/n): ")
        if create_file.lower() != 'y':
            raise ValueError(f"The file {self.config_file} was not created.")
        look_for_path = input(
            'Do you want to automatically search for the "Projets-QCM" directory? (Y/n)')
        if look_for_path.lower() != 'y':
            raise ValueError(f"The file {self.config_file} was not created.")
        home_dir = os.path.expanduser("~")
        projects_dir: str | None = None
        for dir_path, dir_names, filenames in os.walk(home_dir):
            if "Projets-QCM" in dir_names:
                directory = "Projets-QCM"
                projects_dir = os.path.join(dir_path, directory)
                break
            elif "MC-Projects" in dir_names:
                directory = "MC-Projects"
                projects_dir = os.path.join(dir_path, directory)
                break
            else:
                raise ValueError("Could not find the projects directory.")
        print(f"The full path to the 'Projets-QCM' directory is: {projects_dir}")
        configuration = ConfigParser()
        configuration['DEFAULT'] = {
            'projects_dir': projects_dir,
            'student_threshold': 90,
            'company_name': "",
            'company_url': "",
        }
        with open(self.config_file, 'w') as configfile:
            configuration.write(configfile)


class ExamProject:

    def __init__(self, path: str):
        self.projects: str = path  # Path to the Projets-QCM directory
        self.path: str = self._get_path()  # Path to the selected project to analyse
        self.name: str = glob.glob(self.path, recursive=False)[0].split('/')[-1]  # Project name

    def _get_path(self) -> str:
        """
        - Get the list of project directories
        - Presents the list to the user
        - Get the user's selection
        :return: path to the project selected by the user
        """
        if os.path.exists(self.projects):
            subdirectories: list = next(os.walk(self.projects))[1]
            subdirectories.remove('_Archive')
            subdirectories.sort()
            if len(sys.argv) > 1 and sys.argv[1] in subdirectories:
                return os.path.join(self.projects, sys.argv[1])
            return self._user_input(subdirectories)
        raise ValueError(f"The path {self.projects} does not exist.")

    def _user_input(self, sub: list) -> str:
        """
        Presents the list of projects to the user, prompts them to select a project,
        validates the input and returns the path to the selected project.

        :param sub: list of project subdirectories
        :return: path to the selected project
        """
        while True:
            # display numbered list of subdirectories
            print("Here's a list of current projects:")
            for i, directory in enumerate(sub):
                print(f"{i + 1}. {directory}")

            # prompt user to select a subdirectory
            selection: str = input("Enter the number of the project you'd like to select: ")

            # validate user input
            while not selection.isdigit() \
                    or int(selection) not in range(0, len(sub) + 1):
                selection = input(
                    "Invalid input. Enter the number of the project you'd like to select \
                    (type 0 for list): ")

            # If user input is 0, then print the list again
            if selection == '0':
                continue

            # store the path to the selected project
            return os.path.join(self.projects, sub[int(selection) - 1])


class ExamData:

    def __init__(self, path: str, threshold: int = 99):
        self.path: str = path  # Project path
        self.threshold: int = threshold  # bottom limit for calculation of discrimination index
        self.scoring_db: str = os.path.join(self.path, 'data/scoring.sqlite')
        self.capture_db: str = os.path.join(self.path, 'data/capture.sqlite')
        for db in [self.scoring_db, self.capture_db]:
            self._check_db(db)
        with sqlite3.connect(self.scoring_db) as self.conn:
            self.scl: int = self._get_student_code_length()
            self.indicative: pd.DataFrame = pd.read_sql_query("""SELECT DISTINCT question 
                                                                      FROM scoring_question 
                                                                      WHERE indicative = 1""", self.conn)
            self.marks: pd.DataFrame = self._get_marks()
            self.scores: pd.DataFrame = self._get_scores()
            self.variables: pd.DataFrame = pd.read_sql_query("SELECT * FROM scoring_variables",
                                                             self.conn, index_col='name')
            self.answers: pd.DataFrame = self._get_answers()
        with sqlite3.connect(self.capture_db) as self.conn:
            self.capture: pd.DataFrame = self._get_capture()
        self.number_of_examinees: int = self.marks['student'].nunique()
        self.standard_deviation: float = self.marks['mark'].std()
        self.questions: pd.DataFrame = self._get_questions()
        self.items: pd.DataFrame = self._get_items()
        self.general_stats: dict = self._general_stats()
        self.table: pd.DataFrame = self._get_stats_table()

    def _general_stats(self) -> dict:
        return {
            'Number of examinees': self.number_of_examinees,
            NBQ: self.questions['title'].nunique(),
            'Maximum possible mark': float(self.variables['value']['mark_max']),
            'Minimum achieved mark': self.marks['mark'].min(),
            'Maximum achieved mark': self.marks['mark'].max(),
            'Mean': self.marks['mark'].mean(),
            'Median': self.marks['mark'].median(),
            'Mode': self.marks['mark'].mode().iloc[0],
            'Standard deviation': self.standard_deviation,
            'Variance': self.marks['mark'].var(),
            'Standard error of mean': stats.sem(self.marks['mark']),
            'Standard error of measurement': self.standard_deviation / (self.number_of_examinees ** 0.5),
            'Skew': self.marks['mark'].skew(),
            'Kurtosis': self.marks['mark'].kurt(),
        }

    def _get_stats_table(self) -> pd.DataFrame:
        table: pd.DataFrame = pd.DataFrame.from_dict(self.general_stats, orient='index', columns=['Value'])
        table = (table.reset_index(names=['Element', 'Value']).iloc[[0, 2, 3, 4, 5, 6, 7, 8, 12, 13]])
        table['Value'] = table['Value'].apply(pd.to_numeric, errors='coerce')
        return table

    def _get_student_code_length(self) -> int:
        """
        Get the number of student code boxes
        :return: Length of the student code as an integer
        """
        # get the number boxes for the student code, so we can only query the real questions
        scl: int = pd.read_sql_query("""SELECT COUNT(*) FROM scoring_title 
                                     WHERE title LIKE '%student.number%';""",
                                     self.conn).iloc[0].iloc[0]
        return int(scl)

    @staticmethod
    def _check_db(db: str) -> None:
        if not os.path.exists(db):
            print(f"Error: the database {db} does not exist!")
            sys.exit(1)  # terminate the program with an error code

    def _get_marks(self) -> pd.DataFrame:
        pd_mark = pd.read_sql_query("SELECT * FROM scoring_mark", self.conn)
        if pd_mark.empty:
            print("Error: No mark has been recorded in the database")
            sys.exit(1)
        return pd_mark

    def _get_scores(self) -> pd.DataFrame:
        df = pd.read_sql_query(f"""
        SELECT ss.student, ss.question, st.title, ss.score, ss.why, ss.max
        FROM scoring_score ss
        JOIN scoring_title st ON ss.question = st.question
        WHERE ss.question > {self.scl}""", self.conn)
        # Clean the scores to keep track of Cancelled (C), Floored (P), Empty (V) and Error (E)
        # questions
        why = pd.get_dummies(df['why'])
        df = pd.concat([df, why], axis=1)
        df.drop('why', axis=1, inplace=True)
        df.rename(columns={'': 'replied', 'C': 'cancelled', 'P': 'floored', 'V': 'empty',
                           'E': 'error'}, inplace=True)
        df['correct'] = df.apply(lambda row: 1 if row['score'] == row['max'] else 0, axis=1)
        df = df.drop(df[df['question'].isin(self.indicative['question'])].index)
        return df

    def _get_questions(self) -> pd.DataFrame:
        df = self.scores.pivot_table(index=['question', 'title'],
                                     values=self.scores.columns[3:],
                                     aggfunc='sum',
                                     fill_value=0).reset_index()
        df = df.drop(df[df['question'].isin(self.indicative['question'])].index)
        # Get the list of columns to calculate the number of times a question has been presented.
        cols_for_pres = [col for col in ['cancelled', 'empty', 'replied', 'error', 'floored']
                         if col in df.columns]
        df['presented'] = df[cols_for_pres].sum(axis=1)
        # Combined 'replied' and 'floored' as 'replied' otherwise the number of presented is wrong
        # Check if there is a column 'floored' in 'pd_question'
        if 'floored' in df.columns:
            # Create a new column 'replied_new' by summing 'replied' and 'floored'
            df['replied_new'] = df['replied'] + df['floored']
            # Drop the original 'replied' column
            df = df.drop('replied', axis=1)
            # Rename 'replied_new' to 'replied'
            df = df.rename(columns={'replied_new': 'replied'})

        # Get the list of columns for calculate the number of times a question has been replied or
        # left empty...
        cols_for_diff = [col for col in ['floored', 'empty', 'replied', 'error']
                         if col in df.columns]
        # Calculate the difficulty of each question
        df['difficulty'] = df['correct'] / df[cols_for_diff].sum(axis=1)
        # Now the columns are: ['question', 'title', 'cancelled', 'correct', 'empty', 'error', 'max',
        # 'replied', 'score', 'presented', 'difficulty'] - some columns are optional
        # Get item and outcome discrimination if the number of examinees is greater than 99
        if self.number_of_examinees > self.threshold:
            # Create two student dataframes based on the quantile values. They should have the same
            # number of students This should probably be done in a smarter way, outside, in order to
            # be used for item discrimination.
            top_27_df = self.marks.sort_values(by=['mark'], ascending=False).head(
                round(len(self.marks) * 0.27))
            bottom_27_df = self.marks.sort_values(by=['mark'], ascending=False).tail(
                round(len(self.marks) * 0.27))

            df['discrimination'] = self._questions_discrimination(bottom_27_df, top_27_df)

        # Get item (question) correlation
        correlation = self._item_correlation()
        df['correlation'] = df['title'].apply(lambda row: correlation.loc[row]['correlation'])

        return df

    def _get_items(self) -> pd.DataFrame:
        df = self.capture.groupby(['question', 'answer'])['ticked'].sum().reset_index().sort_values(
            by=['question', 'answer'])
        df['correct'] = df.apply(lambda row: self.answers.loc[
            (self.answers['question'] == row['question'])
            & (self.answers['answer'] == row['answer']), 'correct'].values[0], axis=1)
        df = df.merge(self.questions[['question', 'title']], left_on='question', right_on='question')
        df = df[['question', 'title', 'answer', 'correct', 'ticked']].sort_values(
            by=['title', 'answer']).reset_index(drop=True)

        if self.number_of_examinees > self.threshold:
            # Create two student dataframes based on the quantile values. They should have the same
            # number of students This should probably be done in a smarter way, outside, in order to
            # be used for item discrimination.
            top_27_df = self.marks.sort_values(by=['mark'], ascending=False).head(
                round(len(self.marks) * 0.27))
            bottom_27_df = self.marks.sort_values(by=['mark'], ascending=False).tail(
                round(len(self.marks) * 0.27))
            discr = self._items_discrimination(bottom_27_df, top_27_df)
            df = df.merge(discr[['question', 'answer', 'discrimination']], on=['question', 'answer'])
        outcome_correlation: pd.DataFrame = self._outcome_correlation()
        return df.merge(outcome_correlation, on=['question', 'answer'])

    def _get_answers(self) -> pd.DataFrame:
        df = pd.read_sql_query(f"""SELECT DISTINCT question, answer, correct, strategy
                                  FROM scoring_answer 
                                  WHERE question > {self.scl}""", self.conn)
        df = df.drop(df[df['question'].isin(self.indicative['question'])].index)
        df['correct'] = df.apply(lambda x: 1 if (x['correct'] == 1) else 0, axis=1)
        return df

    def _get_capture(self) -> pd.DataFrame:
        df = pd.read_sql_query(f"""
        SELECT student, id_a AS 'question', id_b AS 'answer', total, black, manual 
        FROM capture_zone 
        WHERE type = 4 AND id_a > {self.scl}""", self.conn)
        # Remove rows in pd_capture where question is present in pd_indicative
        df = df.drop(df[df['question'].isin(self.indicative['question'])].index)

        # Apply specific operations to dataframes before returning them
        # pd_capture
        df['ticked'] = df.apply(self._ticked, axis=1)
        return df

    def _ticked(self, row):
        """
        Define if an answer box has been ticked by looking at the darkness of the box compared to the \
        threshold.
        To be used with the capture dataframe to determine if an answer box has been ticked.
        :param row:
        :return: 1 or 0
        """
        # Get thresholds to calculate ticked answers and get the item analysis
        darkness_bottom = float(self.variables.loc['darkness_threshold']['value'])
        darkness_top = float(self.variables.loc['darkness_threshold_up']['value'])

        # If the box has been manually (un-)ticked, set 'ticked' to 1 (ticked) or 0 (un-ticked).
        if row['manual'] != -1:
            return row['manual']
        # If the box darkness is within the threshold => 'ticked' = 1
        elif row['total'] * darkness_bottom < row['black'] <= row['total'] * darkness_top:
            return 1
        else:
            return 0

    def _questions_discrimination(self, bottom: pd.DataFrame, top: pd.DataFrame) -> list:
        """
        Calculate the discrimination index for each question.
        Add a column 'discrimination' to the dataframe 'question_df' with the index for each question
        :return: a list of discrimination indices to be added as a column to question_df
        """
        # Merge questions scores and students mark, bottom quantile
        bottom_merged_df = pd.merge(bottom,
                                    (self.scores.select_dtypes(include=['int64', 'float64'])
                                     if 'cancelled' not in self.scores.columns
                                     else self.scores[self.scores['cancelled'] == 0].
                                     select_dtypes(include=['int64', 'float64'])),
                                    on=['student'], how="inner", validate="many_to_many")

        # Merge questions scores and students mark, top quantile
        top_merged_df = pd.merge(top,
                                 (self.scores.select_dtypes(include=['int64', 'float64'])
                                  if 'cancelled' not in self.scores.columns
                                  else self.scores[self.scores['cancelled'] == 0].
                                  select_dtypes(include=['int64', 'float64'])),
                                 on=['student'], how="inner", validate="many_to_many")

        # Group by question and answer, and calculate the mean mark for each group
        top_mean_df = top_merged_df.groupby(['question', 'student']).mean()
        bottom_mean_df = bottom_merged_df.groupby(['question', 'student']).mean()

        # Calculate the discrimination index for each question
        discrimination = []  # Create a list to store the results
        nb_in_groups = round(len(self.marks) * 0.27)
        for question in top_mean_df.index.levels[0]:
            discr_index = (len(top_mean_df.loc[question][
                                   top_mean_df.loc[question]['score'] == top_mean_df.loc[question][
                                       'score'].max()])
                           - len(bottom_mean_df.loc[question][
                                     bottom_mean_df.loc[question]['score'] ==
                                     bottom_mean_df.loc[question]['score'].max()])) / nb_in_groups
            discrimination.append(discr_index)  # Add the result to the list
        return discrimination

    def _items_discrimination(self, bottom: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the discrimination index for each answer.
        Add a column 'discrimination' to the dataframe 'items_df' with the index for each choice
        :return: a list of discrimination indices to be added as a column to items_df
        """
        # Merge questions scores and students mark, bottom quantile
        bottom_merged_df = bottom.merge(self.capture[['student', 'question', 'answer', 'ticked']],
                                        on='student', how='left')

        # Merge questions scores and students mark, top quantile
        top_merged_df = top.merge(self.capture[['student', 'question', 'answer', 'ticked']],
                                  on='student', how='left')

        # Group by question and answer, and calculate the mean mark for each group
        top_sum_df = top_merged_df[['question', 'answer', 'ticked']].groupby(
            ['question', 'answer']).sum()
        bottom_sum_df = bottom_merged_df[['question', 'answer', 'ticked']].groupby(
            ['question', 'answer']).sum()

        # Calculate the discrimination index for each question
        # Create a dictionary to store the results
        discrimination = {'question': [], 'answer': [], 'discrimination': []}
        nb_in_groups = round(len(self.marks) * 0.27)
        for question in top_sum_df.index.levels[0]:
            for answer in top_sum_df.loc[question].index:
                discr_index = (top_sum_df.loc[question, answer]['ticked']
                               - bottom_sum_df.loc[question, answer]['ticked']) \
                              / nb_in_groups
                discrimination['question'].append(question)
                discrimination['answer'].append(answer)
                discrimination['discrimination'].append(discr_index)
        return pd.DataFrame.from_dict(discrimination, orient='columns')

    def _item_correlation(self):
        """
        Calculate the item correlation for each question.
        :return: a dictionary of item correlations with questions as keys
        """
        if 'cancelled' in self.scores.columns:
            merged_df = pd.merge(self.scores[self.scores['cancelled'] is False], self.marks, on='student',
                                 how="inner", validate="many_to_many")
        else:
            merged_df = pd.merge(self.scores, self.marks, on='student', how="inner", validate="many_to_many")
        item_corr = {}
        questions = merged_df['title'].unique()
        for question in questions:
            item_scores = merged_df.loc[merged_df['title'] == question, 'correct']
            total_scores = merged_df.loc[merged_df['title'] == question, 'mark']
            correlation = stats.pointbiserialr(item_scores, total_scores)
            item_corr[question] = correlation[0]
        return pd.DataFrame.from_dict(item_corr, orient='index', columns=['correlation'])

    def _outcome_correlation(self):
        """
        Calculate the outcome correlation for each outcome of each question.
        :return: a dataframe of outcome correlations
        """
        if 'cancelled' in self.scores.columns:
            merged_df = pd.merge(self.capture, self.scores[['student', 'question', 'cancelled']],
                                 on=['student', 'question'], how="inner", validate="many_to_many")
            merged_df = merged_df[merged_df['cancelled'] is False].merge(self.marks[['student', 'mark']],
                                                                         on='student')
        else:
            merged_df = self.capture.merge(self.marks[['student', 'mark']], on='student')
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
        return pd.DataFrame.from_dict(outcome_corr)


class Charts:

    def __init__(self, exam_project: ExamProject, exam_data: ExamData):
        self.exam: ExamProject = exam_project
        self.data: ExamData = exam_data
        self.image_path: str = os.path.join(self.exam.path, 'img')
        os.makedirs(self.image_path, exist_ok=True)
        self.question_data_columns: list = ['presented', 'cancelled', 'replied', 'correct', 'empty', 'error', ]
        self.actual_data_columns = list(set(self.question_data_columns).intersection(self.data.questions.columns))
        # Calculate the number of bins based on the maximum and minimum marks
        self.mark_bins: int = int(self.data.marks['mark'].max() - self.data.marks['mark'].min())
        self._create_mark_histogram()
        self._create_difficulty_histogram()
        if self.data.number_of_examinees > self.data.threshold:
            self._create_discrimination_histogram()
            self._create_difficulty_vs_discrimination_histogram()
        self._create_question_correlation_histogram()
        self._create_bar_chart()

    def _create_mark_histogram(self) -> None:
        """
        Create a histogram of the 'marks' column in the dataset and save it as an image file.
        """
        # Define constants for plot dimensions and bin count
        plot_width = 9
        plot_height = 4
        bin_count = self.mark_bins

        # Create a subplot with specified dimensions
        plt.subplots(1, 1, figsize=(plot_width, plot_height))

        # Create a histogram of the 'marks' column with KDE and specified bin count
        sns.histplot(self.data.marks['mark'], kde=True, bins=bin_count)

        # Calculate the average value
        average_value: float = self.data.general_stats['Mean']

        # Add a vertical line for the average value
        plt.axvline(average_value, color='red', linestyle='--',
                    label=f'Mean ({round(average_value, 2)})')

        # Set plot labels and legend
        plt.xlabel('Mark')
        plt.ylabel('Number of students')
        plt.legend()

        # Save the plot as an image file
        plt.savefig(os.path.join(self.image_path, 'marks.png'), transparent=False,
                    facecolor='white', bbox_inches="tight")

    def _create_difficulty_histogram(self) -> None:
        """
        Create a histogram of the 'difficulty' column in the dataset and save it as an image file.
        """
        ax = plt.subplots(1, 1, figsize=(9, 4))[1]
        sns.histplot(self.data.questions['difficulty'], bins=30, color='blue', ax=ax)
        average_value = self.data.questions['difficulty'].mean()
        ax.axvline(average_value, color='red', linestyle='--', label=f'Average ({round(average_value, 2)})')
        ax.set_xlabel('Difficulty level (higher is easier)')
        ax.set_ylabel(NBQ)
        ax.legend()
        # Set the color of the bars in the histogram
        threshold1 = 13
        threshold2 = 23
        for patch in ax.patches:
            if patch.get_x() < threshold1:
                patch.set_color('tab:red')
            elif patch.get_x() < threshold2:
                patch.set_color('tab:blue')
            else:
                patch.set_color('tab:green')
        plt.savefig(os.path.join(self.image_path, 'difficulty.png'), transparent=False, facecolor='white',
                    bbox_inches="tight")

    def _create_discrimination_histogram(self) -> None:
        """
        create a histogram of discrimination if enough students
        """
        # Define constants
        bin_count = 30
        average_line_color = 'red'
        histogram_colors = ['tab:orange', 'tab:blue', 'tab:green']

        _, axis = plt.subplots(figsize=(9, 4))  # Set the figure size if desired
        sns.histplot(self.data.questions['discrimination'], bins=bin_count, ax=axis,
                     palette=histogram_colors, label='Discrimination Index')
        average_value = self.data.questions['discrimination'].mean()
        axis.axvline(average_value, color=average_line_color, linestyle='--',
                     label=f'Average ({round(average_value, 2)})')
        axis.set_xlabel('Discrimination index (the higher the better)')
        axis.set_ylabel(NBQ)
        axis.legend()
        plt.savefig(os.path.join(self.image_path, 'discrimination.png'), transparent=False,
                    facecolor='white', bbox_inches="tight")

    def _create_difficulty_vs_discrimination_histogram(self) -> None:
        """
        Create a scatter plot of discrimination index vs difficulty level of all questions.
        If there are fewer than 90 students, no plot is generated.
        The average difficulty and discrimination index are shown as horizontal and vertical lines respectively.

        :raises ValueError: If there is no data or if the required columns are missing from the data.
        """
        fig_size = (9, 4)

        if not self.data.questions.size:
            raise ValueError("No data available")

        required_columns = ['discrimination', 'difficulty']
        if not all(col in self.data.questions.columns for col in required_columns):
            raise ValueError("Missing required columns")

        _, ax = plt.subplots(figsize=fig_size)
        sns.scatterplot(x=self.data.questions['discrimination'], y=self.data.questions['difficulty'], ax=ax)

        average_x = np.nanmean(self.data.questions['discrimination'])
        average_y = np.nanmean(self.data.questions['difficulty'])

        ax.axhline(average_y, color='red', linestyle='--',
                   label=f'Average Difficulty ({average_y:.2f})')
        ax.axvline(average_x, color='blue', linestyle='--',
                   label=f'Average Discrimination ({average_x:.2f})')

        ax.set_xlabel('Discrimination index (the higher the better)')
        ax.set_ylabel('Difficulty level (higher is easier)')
        ax.legend()
        plt.savefig(os.path.join(self.image_path, 'discrimination_vs_difficulty.png'), transparent=False,
                    facecolor='white', bbox_inches="tight")

    def _create_question_correlation_histogram(self) -> None:
        """
        Create a histogram of question correlation
        """
        _, ax3 = plt.subplots(figsize=(9, 4))
        sns.histplot(self.data.questions['correlation'], kde=True, bins=self.mark_bins * 2)
        average_value = self.data.questions['correlation'].mean()
        ax3.axvline(average_value, color='red', linestyle='--',
                    label=f'Average ({average_value:.2f})')
        ax3.set(xlabel='Item correlation', ylabel=NBQ)
        ax3.legend()
        plt.savefig(os.path.join(self.image_path, 'item_correlation.png'), transparent=False,
                    facecolor='white', bbox_inches="tight")

    def _create_bar_chart(self) -> None:
        """
        Create a bar chart for questions data columns
        """
        columns = self.actual_data_columns
        values = self.data.questions[columns].mean().round(2)
        sorted_values = values.sort_values(ascending=False)
        _, ax = plt.subplots(1, 1, figsize=(9, 4))

        sns.barplot(x=sorted_values, y=sorted_values.index, ax=ax)

        ax.set_xlabel('Average Number of Students')
        ax.set_ylabel('Question Status')

        for i, v in enumerate(sorted_values):
            ax.text(v + 3, i, str(v), color='black', ha='center', va='center')

        plt.savefig(os.path.join(self.image_path, 'question_columns.png'),
                    transparent=False, facecolor='white', bbox_inches="tight")


def get_dictionary(dictionary: str) -> dict:
    """
    Get the definitions from the definitions.json file
    :return: a dictionary of definitions
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dictionary + ".json")

    try:
        with open(file_path, "r") as json_file:
            data_dict = json.load(json_file)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {str(e)}")
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
    return data_dict


def get_list_questions(qlist) -> str:
    return ', '.join(qlist[:-1]) + ' and ' + qlist[-1]


def add_blurb_conditionally(df, column, threshold, message_template):
    if column in df.columns:
        filtered_titles = df[df[column] > threshold].sort_values('title')['title'].values
        if filtered_titles.size > 0:
            qlist = get_list_questions(filtered_titles)
            if len(filtered_titles) > 1:
                blb_message = message_template.format(qlist=qlist)
            else:
                blb_message = message_template.format(qlist=filtered_titles[0])
            return blb_message
    return ''


def get_blurb(exam: ExamData):
    """
    Generate a first level of analysis on the performance of the questions. This text can either
    be used as is in the report or passed to ChatGPT for a better wording.
    :return: a string of text describing the data and how to improve the exam questions.
    """
    intro = "According to the data collected, the following questions should probably be reviewed:\n"
    blb = ''

    # Conditions for cancelled questions
    blb += add_blurb_conditionally(
        exam.questions,
        'cancelled',
        exam.questions['presented'] * 0.8,
        "- Questions {qlist} have been cancelled more than 80% of the time.\n"
    )

    # Conditions for empty questions
    blb += add_blurb_conditionally(
        exam.questions,
        'empty',
        exam.questions['presented'] * 0.8,
        "- Questions {qlist} have been empty more than 80% of the time.\n"
    )

    # Conditions for negative discrimination
    if 'discrimination' in exam.questions.columns:
        negative_discrimination = exam.questions[exam.questions['discrimination'] < 0].sort_values('title')[
            'title'].values
        if negative_discrimination.size > 0:
            qlist = get_list_questions(negative_discrimination)
            blb += f"- Questions {qlist} have a negative discrimination, meaning that there is a " \
                   f"possibility of an error in the questions (incorrect outcome indicated as correct).\n"

    # Check for not ticked questions
    if exam.items[exam.items['ticked'] == 0]['title'].values.size > 0:
        not_ticked = exam.items[exam.items['ticked'] == 0]['title'].unique()
        not_ticked.sort()
        if len(not_ticked) > 1:
            qlist = get_list_questions(not_ticked)
            blb += f"- Questions {qlist} have distractors that have never been chosen.\n"
        else:
            blb += f"- Question {not_ticked[0]} has distractors that have never been chosen.\n"

    # Final message construction
    if blb:
        return intro + blb
    else:
        return ("According to the data collected, there are no questions to review based on their "
                "performance.")


def print_dataframes():
    """
    Print the dataframes
    """
    print(f"\nGeneral Statistics:\n{data.table}")
    print(f"\nList of questions (question_df):\n{data.questions.head()}")
    print(f"\nList of answers (answer_df):\n{data.answers.head()}")
    print(f"\nList of items (items_df):\n{data.items.sort_values(by='title').head()}")
    print(f"\nList of variables (variables_df):\n{data.variables}")
    print(f"\nList of capture (capture_df):\n{data.capture.head()}")
    print(f"\nList of mark (mark_df):\n{data.marks.head()}")
    print(f"\nList of score (score_df):\n{data.scores.head()}")


def get_correction_text(df: pd.DataFrame) -> str:
    """
    Generate a paragraphe of text from the capture data to explain the number of boxes ticked or
    un-ticked during the marking process.

    :param df: Dataframe with the capture data
    :type df: pd.Dataframe
    :return: a paragraph of text to display in the report
    :rtype: str
    """
    tres: int = 180
    nb_box_filled = df[(df['manual'] == 1) & (df['black'] < tres)]['student'].count()
    nb_box_emptied = df[(df['manual'] == 0) & (df['black'] > tres)]['student'].count()
    nb_box_untouched = df[(df['manual'] == -1)]['student'].count()
    nb_box_total = df['student'].count()
    nb_box_untouched += df[(df['manual'] == 1) & (df['black'] > tres)]['student'].count()
    nb_box_untouched += df[(df['manual'] == 0) & (df['black'] < tres)]['student'].count()
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
    ic.enable()
    sns.set_theme()
    sns.set_style('darkgrid')
    sns.set_style()
    sns.color_palette("tab10")
    # colour palette (red, green, blue, text color)
    colour_palette: dict = {'heading_1': (23, 55, 83, 255),
                            'heading_2': (109, 174, 219, 55),
                            'heading_3': (40, 146, 215, 55),
                            'white': (255, 255, 255, 0),
                            'yellow': (251, 215, 114, 0),
                            'red': (238, 72, 82, 0),
                            'green': (166, 221, 182, 0),
                            'grey': (230, 230, 230, 0),
                            'blue': (84, 153, 242, 0),
                            }
    config_filename: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG)
    # Get some directory information
    # Get the dir name to check if it matches 'Projets-QCM'
    current_dir_name: str = os.path.basename(os.getcwd())
    # Get the full path in case it matches
    current_full_path: str = os.path.dirname(os.getcwd()) + '/' + current_dir_name
    today = datetime.datetime.now().strftime('%d/%m/%Y')

    config: Settings = Settings(config_filename)
    # Get the Project directory and questions file paths
    project: ExamProject = ExamProject(config.projects)
    data: ExamData = ExamData(project.path)
    definitions: dict = get_dictionary('definitions')
    findings: dict = get_dictionary('findings')
    # OpenAI settings
    ic(data.marks['mark'].kurt())
    ic(data.general_stats)

    blurb: str = ''
    if DEBUG == 0:
        llm = LLM(data.table)
        ic(llm.response)
        blurb = llm.response + '\n\n'
    # Generate the report
    blurb += get_blurb(data)
    correction_text: str = get_correction_text(data.capture)
    ic(blurb)
    ic(correction_text)
    report_params = {
        'project_name': project.name,
        'project_path': project.path,
        'questions': data.questions,
        'items': data.items,
        'stats': data.general_stats,
        'threshold': config.threshold,
        'marks': data.marks,
        'definitions': definitions,
        'findings': findings,
        'palette': colour_palette,
        'blurb': blurb,
        'company_name': config.company_name,
        'company_url': config.company_url,
        'correction': correction_text,
    }
    # plot_charts(report_params)
    charts: Charts = Charts(project, data)
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

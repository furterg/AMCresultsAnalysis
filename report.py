import matplotlib.pyplot as plt
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import json
import seaborn as sns
from fpdf import FPDF
import glob
import datetime

debug = 1
today = datetime.datetime.now().strftime('%d/%m/%Y')
# ### Colour palette
# (r, g, b, text_grey_level)
# colour_palette = {'heading_1': (255, 127, 0), 'heading_2': (254, 162, 57), 'heading_3': (254, 181, 115)}
# colour_palette = {'heading_2': (242, 167, 81), 'heading_1': (141, 104, 61), 'heading_3': (254, 181, 115)}
colour_palette = {'heading_1': (23, 55, 83, 255), 'heading_2': (109, 174, 219, 55), 'heading_3': (40, 146, 215, 55)}


least_performing_prompt = """According to the Classical Test Theory, which are the least performing questions? Please 
take into account all the columns and write a paragraph to explain your choice. Don't mention the Classical Test 
Theory in your reply."""
most_performing_prompt = """According to the Classical Test Theory, which are the best performing questions? Please
take into account all the columns and write a paragraph to explain your choice. Don't mention the Classical Test 
Theory in your reply."""

dummy_gpt_reply = """The least performing questions according to the data are Q003, Q001, Q048, Q046, and Q002. These \
questions have the lowest values in each column, such as cancelled, correct, empty, floored, max, replied, score, \
presented, difficulty, discrimination, and correlation.\nThe best performing questions according to the data are \
Q031, Q030, Q021, Q043, and Q029. These questions have the highest correlation values, indicating that they are the \
 most reliable and valid questions."""


# def get_definitions():
#     """
#     Get the definitions from the definitions.json file
#     :return: a dictionary of definitions
#     """
#     file_path = "definitions.json"
#
#     try:
#         with open(file_path, "r") as json_file:
#             data = json.load(json_file)
#     except FileNotFoundError:
#         print(f"File '{file_path}' not found.")
#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON file: {str(e)}")
#     except Exception as e:
#         print(f"Error loading JSON file: {str(e)}")
#     return data


def to_letter(value):
    """
    Convert a value to a letter
    :param value: value to convert (int)
    :return: letter corresponding to the value (str)
    """
    letter = chr(ord('A') + value - 1)
    return letter


class PDF(FPDF):
    def __init__(self, project_name):
        super().__init__()
        # Margin
        self.margin = 10
        # Page width: Width of A4 is 210mm
        self.pw = 210 - 2 * self.margin
        # Cell height
        self.ch = 6
        self.project = project_name

    def header(self):
        self.set_font('Helvetica', '', 12)
        self.cell(w=0, h=8, txt=f'{self.project} - Exam report', border=0, ln=1, align='C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', '', 12)
        self.cell(w=self.pw / 3, h=8,
                  txt=f"©{datetime.datetime.now().strftime('%Y')} - Print&Scan",
                  border=0, ln=0, align='L')
        self.cell(self.pw / 3, 8, f'Page {self.page_no()}', 0, 0, 'C')
        self.cell(w=self.pw / 3, h=8, txt=f"www.printandscan.fr", border=0, ln=0, align='R')

    def set_bg(self, colour):
        self.set_fill_color(colour_palette[colour][0], colour_palette[colour][1], colour_palette[colour][2])
        self.set_text_color(colour_palette[colour][3])


def generate_pdf_report(params):
    """
    https://towardsdatascience.com/how-to-create-a-pdf-report-for-your-data-analysis-in-python-2bea81133b
    The ln parameter defines where the position should go after this cell:
    0: to the right of the current cell
    1: to the beginning of the next line
    2: below the current cell
    :return:
    """
    project_name = params['project_name']
    stats = params['stats']
    questions = params['questions']
    items = params['items']
    author = params['author']
    threshold = params['threshold']
    definitions = params['definitions']
    blob = params['blob']

    question_data_columns = ['presented', 'cancelled', 'replied', 'correct', 'empty', 'error', ]
    question_analysis_columns = ['difficulty', 'discrimination', 'correlation', ]
    outcome_data_columns = ['answer', 'correct', 'ticked', 'discrimination', ]

    pdf = PDF(project_name)
    ch = pdf.ch
    pw = pdf.pw
    pdf.add_page()
    pdf.image('./logo.png',
              x=pw / 2 - 10, y=None, w=40, h=0, type='PNG')
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(w=0, h=12, txt=f"{project_name} - Examination report", ln=1, align='C')
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(w=pw / 2, h=6, txt=f"Author: {author}", ln=0, align='L')
    pdf.cell(w=pw / 2, h=6, txt=f"Date: {today}", ln=1, align='R')
    pdf.ln(ch)
    pdf.set_bg('heading_2')
    # pdf.set_fill_color(190, 192, 191)
    # First row of data
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(w=pw / 5, h=6, txt="Number of", ln=0, align='C', fill=True, border='LTR')
    pdf.cell(w=pw / 5, h=6, txt="Number of", ln=0, align='C', fill=True, border='LTR')
    pdf.cell(w=pw / 5, h=6, txt="Maximum", ln=0, align='C', fill=True, border='LTR')
    pdf.cell(w=pw / 5, h=6, txt="Minimum", ln=0, align='C', fill=True, border='LTR')
    pdf.cell(w=pw / 5, h=6, txt="Maximum", ln=1, align='C', fill=True, border='LTR')
    pdf.cell(w=pw / 5, h=6, txt="examinees", ln=0, align='C', fill=True, border='LBR')
    pdf.cell(w=pw / 5, h=6, txt="questions", ln=0, align='C', fill=True, border='LBR')
    pdf.cell(w=pw / 5, h=6, txt="possible mark", ln=0, align='C', fill=True, border='LBR')
    pdf.cell(w=pw / 5, h=6, txt="possible mark", ln=0, align='C', fill=True, border='LBR')
    pdf.cell(w=pw / 5, h=6, txt="achieved mark", ln=1, align='C', fill=True, border='LBR')
    pdf.set_font('Helvetica', '', 12)
    for key in ['Number of examinees', 'Number of questions', 'Maximum possible mark',
                'Minimum achieved mark', 'Maximum achieved mark']:
        text = str(stats.loc[key]['Value'])
        pdf.cell(w=pw / 5, h=ch, txt=text, ln=0, align='C', border='LBR')
    pdf.ln(ch)
    # Second row of data
    pdf.set_font('Helvetica', 'B', 12)
    for text in ['Mean', 'Median', 'Mode', 'Std Dev', 'Variance']:
        pdf.cell(w=pw / 5, h=6, txt=text, ln=0, align='C', fill=True, border=1)
    pdf.ln(ch)
    pdf.set_font('Helvetica', '', 12)
    for key in ['Mean', 'Median', 'Mode', 'Standard deviation', 'Variance']:
        text = str(round(stats.loc[key]['Value'], 2))
        pdf.cell(w=pw / 5, h=ch, txt=text, ln=0, align='C', border='LBR')
    pdf.ln(ch)
    # Third row of data
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(w=pw / 5, h=6, txt="Std error", ln=0, align='C', fill=True, border='LTR')
    pdf.cell(w=pw / 5, h=6, txt="Std error", ln=0, align='C', fill=True, border='LTR')
    pdf.cell(w=pw / 5, h=6, txt="Skew", ln=0, align='C', fill=True, border='LTR')
    pdf.cell(w=pw / 5, h=6, txt="Kurtosis", ln=0, align='C', fill=True, border='LTR')
    pdf.cell(w=pw / 5, h=6, txt="Test", ln=1, align='C', fill=True, border='LTR')
    pdf.cell(w=pw / 5, h=6, txt="of mean", ln=0, align='C', fill=True, border='LBR')
    pdf.cell(w=pw / 5, h=6, txt="of measurement", ln=0, align='C', fill=True, border='LBR')
    pdf.cell(w=pw / 5, h=6, txt="", ln=0, align='C', fill=True, border='LBR')
    pdf.cell(w=pw / 5, h=6, txt="", ln=0, align='C', fill=True, border='LBR')
    pdf.cell(w=pw / 5, h=6, txt="reliability", ln=1, align='C', fill=True, border='LBR')
    pdf.set_font('Helvetica', '', 12)
    for key in ['Standard error of mean', 'Standard error of measurement', 'Skew', 'Kurtosis',
                'Test reliability (Cronbach\'s Alpha)']:
        text = str(round(stats.loc[key]['Value'], 2)) \
            if key != 'Test reliability (Cronbach\'s Alpha)' \
            else str(round(stats.loc[key]['Value'][0], 6))
        pdf.cell(w=pw / 5, h=ch, txt=text, ln=0, align='C', border='LBR')
    pdf.ln(ch + 3)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(w=pw / 2, h=6, txt="Frequency of marks", ln=0, align='C', fill=True, border=1)
    x = pdf.get_x()
    pdf.cell(w=pw / 2, h=6, txt="Difficulty levels", ln=1, align='C', fill=True, border=1)
    y = pdf.get_y()
    pdf.image('./img/marks.png', w=pw / 2, type='PNG')
    pdf.image('./img/difficulty.png', w=pw / 2, x=x, y=y, type='PNG')
    # pdf.ln(ch)
    if stats.loc['Number of examinees'][0] > threshold:
        pdf.cell(w=pw / 2, h=6, txt="Question discrimination", ln=0, align='C', fill=True, border=1)
        x = pdf.get_x()
        pdf.cell(w=pw / 2, h=6, txt="Difficulty vs Discrimination", ln=1, align='C', fill=True, border=1)
        y = pdf.get_y()
        pdf.image('./img/discrimination.png', w=pw / 2, type='PNG')
        pdf.image('./img/discrimination_vs_difficulty.png', w=pw / 2, x=x, y=y, type='PNG')
    pdf.cell(w=pw / 2, h=6, txt="Question correlation", ln=0, align='C', fill=True, border=1)
    x = pdf.get_x()
    pdf.cell(w=pw / 2, h=6, txt="Outcome correlation", ln=1, align='C', fill=True, border=1)
    pdf.set_font('Helvetica', 'B', 16)
    y = pdf.get_y()
    pdf.image('./img/item_correlation.png', w=pw / 2, type='PNG')
    pdf.image('./img/outcome_correlation.png', w=pw / 2, x=x, y=y, type='PNG')
    pdf.ln(ch)
    pdf.cell(w=pw, h=6, txt="TL;DR", ln=1, align='L ', fill=False, border=0)
    pdf.set_font('Helvetica', '', 12)

    # Use OpenAI API to analyse the question dataframe and explain the least and most performing questions
    if debug == 1:
        txt = blob
        print('blob is set')
    else:
        question_agent = create_pandas_dataframe_agent(OpenAI(temperature=0, max_tokens=512),
                                                       question_df, verbose=False)
        txt = question_agent.run(least_performing_prompt)
        txt += "\n\n" + question_agent.run(most_performing_prompt)
        print('GPT is used')
    pdf.multi_cell(w=pw, h=ch, txt=txt)
    # print(txt)

    q_data_columns = []
    q_analysis_columns = []
    o_data_columns = []
    for col in question_data_columns:
        if (col in questions.columns) and (questions[col].std() != 0):
            q_data_columns.append(col)
    for col in question_analysis_columns:
        if (col in questions.columns) and (questions[col].std() != 0):
            q_analysis_columns.append(col)
    for col in outcome_data_columns:
        if (col in items.columns) and (items[col].std() != 0):
            o_data_columns.append(col)
    cols_1 = pw / len(q_data_columns)
    cols_2 = pw / len(q_analysis_columns)
    cols_3 = pw / len(o_data_columns)
    for question in questions.sort_values('title')['title'].values:
        nb_presented = questions[questions['title'] == question]['presented'].values[0] \
            if 'presented' in q_data_columns else stats.loc['Number of examinees']['Value']
        pdf.ln(ch / 2)
        if pdf.get_y() > 250:
            pdf.add_page()
        pdf.set_bg('heading_1')
        # pdf.set_fill_color(95, 94, 94)
        pdf.cell(w=pw, h=ch, txt=f"Question - {question}", ln=1, align='C', fill=True, border=1)
        pdf.set_bg('heading_2')
        # pdf.set_fill_color(190, 192, 191)
        # First row of data
        for col in q_data_columns:
            pdf.cell(w=cols_1, h=ch, txt=f"{col}", ln=0, align='C', fill=True, border=1)
        pdf.ln(ch)
        for col in q_data_columns:
            value = questions[questions['title'] == question][col].values[0]
            txt = f"{value}" if (value == nb_presented) or (value == 0) \
                else f"{value} ({round(value / nb_presented * 100, 2)}%)"
            pdf.cell(w=cols_1, h=ch, txt=txt, ln=0, align='C', fill=False, border=1)
        pdf.ln(ch)
        # Second row of data
        for col in q_analysis_columns:
            pdf.cell(w=cols_2, h=ch, txt=f"{col}", ln=0, align='C', fill=True, border=1)
        pdf.ln(ch)
        for col in q_analysis_columns:
            value = questions[questions['title'] == question][col].values[0]
            label = ''
            if col == 'difficulty':
                if value <= 0.4:
                    label = 'difficult'
                elif value <= 0.6:
                    label = 'intermediate'
                else:
                    label = 'easy'
            elif col == 'discrimination':
                if value < 0:
                    label = 'To be reviewed!'
                elif value <= 0.16:
                    label = 'Low'
                elif value <= 0.3:
                    label = 'Moderate'
                elif value <= 0.5:
                    label = 'High'
                else:
                    label = 'Very high'
            elif col == 'correlation':
                if value < 0:
                    label = 'To be reviewed!'
                elif value <= 0.1:
                    label = 'None'
                elif value <= 0.2:
                    label = 'Low'
                elif value <= 0.3:
                    label = 'Moderate'
                elif value <= 0.45:
                    label = 'Strong'
                else:
                    label = 'Very strong'
            txt = f"{round(value, 4)} ({label})"
            pdf.cell(w=cols_2, h=ch, txt=txt, ln=0, align='C', fill=False, border=1)
        pdf.ln(ch)
        # Breakdown of question's outcomes
        if items.loc[(items['title'] == question), 'answer'].count() > 10:
            continue
        for col in o_data_columns:
            pdf.cell(w=cols_3, h=ch, txt=f"{col}", ln=0, align='C', fill=True, border=1)
        else:
            for answer in items.loc[items['title'] == question, 'answer'].values:
                pdf.ln(ch)
                for col in o_data_columns:
                    value = items.loc[(items['title'] == question) & (items['answer'] == answer), col].values[0]
                    if col == 'answer':
                        txt = to_letter(value)
                    elif col == 'correct':
                        txt = '*' if value == 1 else ''
                    elif col == 'ticked':
                        txt = str(round(value)) + ' (' + str(round((value / nb_presented * 100), 2)) + '%)'
                    elif col == 'discrimination':
                        if items.loc[(items['title'] == question)
                                        & (items['answer'] == answer), 'correct'].values[0] == 0:
                            label = '-'
                        elif value < 0:
                            label = 'To be reviewed!'
                        elif value <= 0.16:
                            label = 'Low'
                        elif value <= 0.3:
                            label = 'Moderate'
                        elif value <= 0.5:
                            label = 'High'
                        else:
                            label = 'Very high'
                        txt = label if label == '-' else str(round(value, 4)) + ' (' + label + ')'
                    else:
                        pass
                    pdf.cell(w=cols_3, h=ch, txt=txt, ln=0, align='C', fill=False, border=1)
        pdf.ln(ch)

    if pdf.get_y() > 100:
        pdf.add_page()
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(w=pw, h=ch, txt='Definitions', ln=1, align='C', fill=False, border=0)
    pdf.ln(ch - 3)
    for key in definitions:
        pdf.ln(ch)
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(w=pw, h=ch, txt=key, ln=1, align='L', fill=False, border=0)
        pdf.set_font('helvetica', '', 12)
        pdf.multi_cell(w=pw, h=ch, txt=definitions[key], align='L', fill=False, border=0)
    # pdf.multi_cell(w=0, h=5, txt='some text here')
    # pdf.multi_cell(w=0, h=5, txt='some other text')

    pdf.output(f'./example.pdf', 'F')

    pdf.set_bg('heading_1')
    # pdf.set_fill_color(95, 94, 94)


def plot_charts(params):
    marks = params['marks']
    items = params['items']
    questions = params['questions']
    stats = params['stats']
    threshold = params['threshold']

    # Calculate the number of bins based on the maximum and minimum marks
    mark_bins = int(float(stats.loc['Maximum achieved mark', 'Value'])
                    - float(stats.loc['Minimum achieved mark', 'Value']))

    # create a histogram of the 'mark' column
    marks_plot, ax1 = plt.subplots(1, 1, figsize=(9, 4))
    sns.histplot(marks['mark'], kde=True, bins=mark_bins)
    # ax.set_title('Frequency of Marks')
    plt.savefig('./img/marks.png',
                transparent=False,
                facecolor='white',
                bbox_inches="tight")

    # create a histogram of the 'mark' column
    diff_plot, ax2 = plt.subplots(1, 1, figsize=(9, 4))
    sns.histplot(questions['difficulty'], bins=30, color='blue')
    ax2.set_xlabel('Difficulty level (higher is easier)')
    ax2.set_ylabel('Number of questions')
    # Set the color of the bars in the first histogram
    for patch in ax2.patches[:13]:
        patch.set_color('tab:red')
    for patch in ax2.patches[13:23]:
        patch.set_color('tab:blue')
    for patch in ax2.patches[23:]:
        patch.set_color('tab:green')
    plt.savefig('./img/difficulty.png', transparent=False, facecolor='white', bbox_inches="tight")

    # create a histogram of discrimination if enough students
    if stats.loc['Number of examinees'][0] > threshold:
        fig, ax = plt.subplots(figsize=(9, 4))  # Set the figure size if desired
        sns.histplot(questions['discrimination'], bins=30, ax=ax, color='orange')
        ax.set_xlabel('Discrimination index (the higher the better)')
        ax.set_ylabel('Number of questions')
        # Set the color of the bars in the second histogram
        for patch in ax.patches[:13]:
            patch.set_color('tab:orange')
        for patch in ax.patches[13:23]:
            patch.set_color('tab:blue')
        for patch in ax.patches[23:]:
            patch.set_color('tab:green')
        plt.savefig('./img/discrimination.png', transparent=False, facecolor='white', bbox_inches="tight")

        # Plot difficulty vs discrimination
        fig, ax = plt.subplots(figsize=(9, 4))  # Set the figure size if desired
        sns.scatterplot(x=questions['discrimination'], y=questions['difficulty'], ax=ax)
        ax.set_xlabel('Discrimination index (the higher the better)')
        ax.set_ylabel('Difficulty level (higher is easier)')
        plt.savefig('./img/discrimination_vs_difficulty.png', transparent=False, facecolor='white', bbox_inches="tight")

    # create a histogram of question correlation
    itemcorr_plot, ax3 = plt.subplots(1, 1, figsize=(9, 4))
    sns.histplot(questions['correlation'], kde=True, bins=mark_bins * 2)
    ax3.set_xlabel('Question correlation')
    # ax.set_title('Frequency of Marks')
    plt.savefig('./img/item_correlation.png', transparent=False, facecolor='white', bbox_inches="tight")

    # create a histogram of question correlation
    outcomecorr_plot, ax4 = plt.subplots(1, 1, figsize=(9, 4))
    sns.histplot(items[items['correct'] == 1]['correlation'], kde=True, bins=mark_bins * 2)
    ax4.set_xlabel('Outcome correlation (correct outcomes only)')
    # ax.set_title('Frequency of Marks')
    plt.savefig('./img/outcome_correlation.png', transparent=False, facecolor='white', bbox_inches="tight")


if __name__ == '__main__':
    # sns.color_palette("tab10")
    # sns.color_palette("rocket")
    directory_path = "/Users/greg/Dropbox/01-QCM/_AMC/Projets-QCM"
    capture_path = directory_path + '/202304-5A1D01_01/data/capture.sqlite'
    scoring_path = directory_path + '/202304-5A1D01_01/data/scoring.sqlite'
    amcProject = directory_path + '/202304-5A1D01_01'
    amcProject_name = glob.glob(amcProject, recursive=False)[0].split('/')[-1]

    # loading dataframes from disk to be used in tests
    question_df = pd.read_pickle('./question_df.pkl')
    answer_df = pd.read_pickle('./answer_df.pkl')
    items_df = pd.read_pickle('./items_df.pkl')
    variables_df = pd.read_pickle('./variables_df.pkl')
    capture_df = pd.read_pickle('./capture_df.pkl')
    mark_df = pd.read_pickle('./mark_df.pkl')
    score_df = pd.read_pickle('./score_df.pkl')
    stats_df = pd.read_pickle('./stats_df.pkl')

    report_params = {
        'project_name': amcProject_name,
        'questions': question_df,
        'items': items_df,
        'author': 'Gregory Furter',
        'stats': stats_df,
        'threshold': 99,
        'marks': mark_df,
        'definitions': 'Dummy definitions',
        'blob': 'Dummy blob',
    }

    plot_charts(report_params)

    generate_pdf_report(report_params)

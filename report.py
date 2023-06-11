import matplotlib.pyplot as plt
import os
import seaborn as sns
from fpdf import FPDF, TitleStyle
import datetime
from tabulate import tabulate

today = datetime.datetime.now().strftime('%d/%m/%Y')

explanations = {
    'top_discrimination': {
        'column': 'discrimination',
        'limit': 0.7,
        'heading': 'Top Discrimination Index',
        'text': f"""
Questions with a discrimination index above {{limit}} effectively differentiate between individuals with varying levels of the measured construct, making them valuable for future tests. These questions contribute to the reliability and validity of the assessment by accurately ranking and measuring individual differences. Retaining high-discriminating questions helps maintain the test's difficulty level and ensures it remains a robust tool for assessing knowledge or skills.

The discrimination index measures the item's ability to distinguish between high and low performers on a test. It is calculated by comparing the proportion of examinees of the top 27% of participants (in terms of assessment score) minus the lowest 27% on a particular item. A discrimination index above {{limit}} indicates strong discriminative power, reflecting the item's effectiveness in differentiating between individuals with varying levels of the construct being measured.

The following **{{questions}} (out of {{total}})** item{{plural}} a high discriminaiton index, this represents **{{percent}}%** of all the questions:
"""},
    'negative_discrimination': {
        'column': 'discrimination',
        'limit': 0,
        'heading': 'Negative Discrimination Index',
        'text': f"""
The discrimination index, also known as the item discrimination, measures the ability of an item to differentiate between high-scoring and low-scoring individuals on a test.

A negative discrimination index for a test item indicates that individuals who scored higher on the overall test performed worse on that particular item compared to individuals who scored lower on the overall test. In other words, individuals who are more knowledgeable or proficient in the construct being measured by the test are more likely to answer the item incorrectly.

A negative discrimination index suggests that the item is "reverse-scored" or "keyed negatively" compared to the other items in the test. It means that the item is counterintuitive or confusing to test takers, as higher-scoring individuals are more likely to select incorrect responses, while lower-scoring individuals are more likely to select correct responses.

In practical terms, a negative discrimination index indicates that the item may be problematic and should be carefully reviewed. It may suggest the need to revise the item or consider removing it from the test if it is not effectively measuring the construct of interest. Negative discrimination can adversely affect the reliability and validity of a test, as it undermines the test's ability to accurately rank individuals according to their level of the measured trait or construct.

The following **{{questions}} (out of {{total}})** item{{plural}} a negative discrimination index, this represents **{{percent}}%** of all the questions:

"""},
    'low_correlation': {
        'column': 'correlation',
        'limit': 0.2,
        'heading': 'Low Correlation Index',
        'text': f"""
The following questions may need to be checked as they have a low correlation index. A low point biserial correlation indicates a weak relationship between the item response and the total test score.

Specifically, a low point biserial correlation suggests that the item is not effectively discriminating between examinees who perform well on the test and those who perform poorly. It indicates that the item is not differentiating between individuals with higher and lower levels of the latent trait or construct being measured by the test.

In practical terms, a low point biserial correlation implies that the item may not be a good indicator of the test takers' overall ability or proficiency in the tested domain. It may suggest that the item needs to be reviewed or revised to improve its ability to discriminate between individuals with different levels of the construct..

The following **{{questions}} (out of {{total}})** item{{plural}} a low correlation index, this represents **{{percent}}%** of all the questions:
"""},
                }


def to_letter(value):
    """
    Convert a value to a letter
    :param value: value to convert (int)
    :return: letter corresponding to the value (str)
    """
    letter = chr(ord('A') + value - 1)
    return letter


class PDF(FPDF):
    def __init__(self, project_name, colour_palette, name, url):
        super().__init__()
        # Margin
        self.margin = 10
        # Page width: Width of A4 is 210mm
        self.pw = 210 - 2 * self.margin
        # Cell height
        self.ch = 6
        self.project = project_name
        self.colour_palette = colour_palette
        self.name = name
        self.url = url

    def header(self):
        self.set_font('Helvetica', '', 12)
        self.cell(w=self.pw / 2, h=6, txt=f'{self.project} - Exam report', border=0, ln=0,
                  align='L')
        self.cell(w=self.pw / 2, h=6, txt=f"{today}", ln=1, align='R')

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', '', 12)
        self.cell(w=self.pw / 3, h=8,
                  txt=f"©{datetime.datetime.now().strftime('%Y')} - {self.name}",
                  border=0, ln=0, align='L')
        self.cell(self.pw / 3, 8, f'Page {self.page_no()}', 0, 0, 'C')
        self.cell(w=self.pw / 3, h=8, txt=self.url, border=0, ln=0, align='R')

    def set_bg(self, colour):
        self.set_fill_color(self.colour_palette[colour][0],
                            self.colour_palette[colour][1],
                            self.colour_palette[colour][2])
        self.set_text_color(self.colour_palette[colour][3])


def p(pdf, text, **kwargs):
    "Inserts a paragraph"
    pdf.multi_cell(
        w=pdf.pw,
        h=pdf.font_size,
        txt=text,
        new_x="LMARGIN",
        new_y="NEXT",
        **kwargs,
    )


def render_toc(pdf, outline):
    pdf.y += 25
    pdf.set_font("Helvetica", size=16)
    pdf.underline = True
    pdf.x = pdf.l_margin
    p(pdf, "Table of contents:", align="C")
    pdf.underline = False
    pdf.y += 15
    pdf.set_font("Courier", size=14)
    last_page_digits = len(str(outline[-1].page_number))
    for section in outline:
        link = pdf.add_link(page=section.page_number)
        p(
            pdf,
            f'{" " * section.level * 2} {section.name} {"." * (56 - section.level * 2 - len(section.name))} {" " * (last_page_digits - len(str(section.page_number)))}{section.page_number}',
            align="L",
            link=link,
        )
        pdf.y += 6


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
    threshold = params['threshold']
    definitions = params['definitions']
    blurb = params['blurb']
    palette = params['palette']
    report_path = params['project_path']
    image_path = report_path + '/img'
    report_file_path = report_path + '/' + project_name + '-report.pdf'
    company = params['company_name']
    url = params['company_url']

    question_data_columns = ['presented', 'cancelled', 'replied', 'correct', 'empty', 'error', ]
    actual_data_columns = list(set(question_data_columns).intersection(questions.columns))
    question_analysis_columns = ['difficulty', 'discrimination', 'correlation', ]
    actual_analysis_columns = list(set(question_analysis_columns).intersection(questions.columns))

    outcome_data_columns = ['answer', 'correct', 'ticked', 'discrimination', ]
    actual_outcome_columns = list(set(outcome_data_columns).intersection(items.columns))

    pdf = PDF(project_name, palette, company, url)
    ch = pdf.ch
    pw = pdf.pw
    pdf.set_section_title_styles(
        # Level 0 titles:
        TitleStyle(
            font_family="Helvetica",
            font_style="B",
            font_size_pt=24,
            color=palette['heading_1'][:3],
            underline=True,
            t_margin=5,
            l_margin=10,
            b_margin=2,
        ),
        # Level 1 subtitles:
        TitleStyle(
            font_family="Helvetica",
            font_style="B",
            font_size_pt=20,
            color=palette['heading_2'][:3],
            underline=True,
            t_margin=5,
            l_margin=20,
            b_margin=2,
        ),
    )
    pdf.add_page()
    if not os.path.isfile('./logo.png'):
        pdf.cell(w=pw, h=ch, txt="Your logo goes here. Place a 'logo.png' in the same folder",
                 border=0, ln=1, align='C')
    pdf.image('./logo.png', x=pw / 2 - 10, y=None, w=40, h=0, type='PNG')
    pdf.set_y(50)
    pdf.set_font(size=18)
    pdf.cell(w=pw, h=ch, txt=f"{project_name} - Examination report", align="C")
    pdf.set_font(size=12)
    pdf.insert_toc_placeholder(render_toc)

    pdf.set_font('Helvetica', 'B', 16)
    pdf.start_section("General Statistics")
    pdf.ln(ch / 2)
    pdf.set_bg('heading_2')

    # First row of data
    pdf.set_font('Helvetica', 'B', 12)
    for txt in ['Number of', 'Number of', 'Maximum', 'Minimum', 'Maximum']:
        pdf.cell(w=pw / 5, h=6, txt=txt, ln=0, align='C', fill=True, border='LTR')
    pdf.ln()
    for txt in ['examinees', 'questions', 'possible mark', 'achieved mark', 'achieved mark']:
        pdf.cell(w=pw / 5, h=6, txt=txt, ln=0, align='C', fill=True, border='LBR')
    pdf.ln()
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
    for txt in ['Std error', 'Std error', 'Skew', 'Kurtosis', 'Average']:
        pdf.cell(w=pw / 5, h=6, txt=txt, ln=0, align='C', fill=True, border='LTR')
    pdf.ln()
    for txt in ['of mean', 'of measurement', '', '', 'Difficulty']:
        pdf.cell(w=pw / 5, h=6, txt=txt, ln=0, align='C', fill=True, border='LBR')
    pdf.ln()
    stats.loc['Average Difficulty'] = questions['difficulty'].mean()
    pdf.set_font('Helvetica', '', 12)
    for key in ['Standard error of mean', 'Standard error of measurement', 'Skew', 'Kurtosis',
                'Average Difficulty']:
        text = str(round(stats.loc[key]['Value'], 2))
        pdf.cell(w=pw / 5, h=ch, txt=text, ln=0, align='C', border='LBR')
    pdf.ln(ch + 3)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(w=pw / 2, h=6, txt="Frequency of marks", ln=0, align='C', fill=True, border=1)
    x = pdf.get_x()
    pdf.cell(w=pw / 2, h=6, txt="Difficulty levels", ln=1, align='C', fill=True, border=1)
    y = pdf.get_y()
    pdf.image(image_path + '/marks.png', w=pw / 2, type='PNG')
    pdf.image(image_path + '/difficulty.png', w=pw / 2, x=x, y=y, type='PNG')
    if stats.loc['Number of examinees'][0] > threshold:
        pdf.cell(w=pw / 2, h=6, txt="Item discrimination", ln=0, align='C', fill=True, border=1)
        x = pdf.get_x()
        pdf.cell(w=pw / 2, h=6, txt="Difficulty vs Discrimination", ln=1, align='C', fill=True,
                 border=1)
        y = pdf.get_y()
        pdf.image(image_path + '/discrimination.png', w=pw / 2, type='PNG')
        pdf.image(image_path + '/discrimination_vs_difficulty.png', w=pw / 2, x=x, y=y, type='PNG')
    pdf.cell(w=pw / 2, h=6, txt="Item correlation", ln=0, align='C', fill=True, border=1)
    x = pdf.get_x()
    pdf.cell(w=pw / 2, h=6, txt="Average Answering", ln=1, align='C', fill=True, border=1)
    y = pdf.get_y()
    pdf.image(image_path + '/item_correlation.png', w=pw / 2, type='PNG')
    pdf.image(image_path + '/question_columns.png', w=pw / 2, x=x, y=y, type='PNG')
    pdf.add_page()
    pdf.ln(ch)
    pdf.start_section("Findings")
    pdf.start_section("Summary (TL;DR)", level=1)
    pdf.set_font('Helvetica', '', 12)
    pdf.multi_cell(w=pw, h=ch, txt=blurb, markdown=True)
    pdf.set_bg('white')
    # Top performing questions
    if 'discrimination' in questions.columns \
            and questions[questions['discrimination'] > explanations['top_discrimination']['limit']].shape[0] > 0:
        limit = explanations['top_discrimination']['limit']
        pdf.start_section(f"{explanations['top_discrimination']['heading']}", level=1)
        nb_questions = questions[questions['discrimination'] > limit].shape[0]
        plural = 's have' if nb_questions> 1 else ' has'
        txt = explanations['top_discrimination']['text'] \
            .format(questions=nb_questions,
                    total=questions.shape[0],
                    plural=plural,
                    limit=limit,
                    percent=str(round(
                        100 * nb_questions / questions.shape[0], 2)))
        pdf.multi_cell(w=pw, txt=txt, markdown=True, ln=1)
        data = questions[questions['discrimination'] > limit][
            ['title', 'correct', 'empty', 'difficulty', 'discrimination']].sort_values(by='discrimination', ascending=False)
        pdf.write_html(data.to_html(index=False))
    # question, correct, empty, difficulty, discrimination
    if 'discrimination' in questions.columns \
            and questions[questions['discrimination'] < 0].shape[0] > 0:
        nb_questions = questions[questions['discrimination'] < 0].shape[0]
        plural = 's have' if nb_questions > 1 else ' has'
        pdf.set_font('Helvetica', '', 12)
        pdf.start_section(f"{explanations['negative_discrimination']['heading']}", level=1)
        txt = explanations['negative_discrimination']['text'] \
            .format(questions=nb_questions,
                    total=questions.shape[0],
                    plural=plural,
                    percent=str(round(
                        100 * nb_questions / questions.shape[0], 2)))
        pdf.multi_cell(w=pw, txt=txt, markdown=True, ln=1)
        data = questions[questions['discrimination'] < 0][
            ['title', 'correct', 'empty', 'difficulty', 'discrimination']].sort_values(
            'discrimination')
        pdf.write_html(data.to_html(index=False))
    if 'correlation' in questions.columns \
            and questions[questions['correlation'] < 0.2].shape[0] > 0:
        pdf.start_section("Low Correlation Index", level=1)
        nb_questions = questions[questions['correlation'] < 0.2].shape[0]
        plural = 's have' if nb_questions > 1 else ' has'
        txt = explanations['low_correlation']['text'] \
            .format(questions= nb_questions,
                    total=questions.shape[0],
                    plural=plural,
                    percent=str(round(
                        100 * nb_questions / questions.shape[0], 2)))
        pdf.multi_cell(w=pw, txt=txt, markdown=True, ln=1)
        data = questions[questions['correlation'] < 0.2][
            ['title', 'correct', 'empty', 'difficulty', 'correlation']].sort_values('correlation')
        pdf.write_html(data.to_html(index=False))

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
    cols_1 = pw / len(q_data_columns)  # Presented, cancelled...
    cols_2 = pw / len(q_analysis_columns)  # Difficulty, Discrimination...
    cols_3 = pw / len(o_data_columns)  # Answer, Correct, Ticked...
    pdf.start_section("Items and Outcomes (detailed)", level=1)
    for question in questions.sort_values('title')['title'].values:
        nb_presented = questions[questions['title'] == question]['presented'].values[0] \
            if 'presented' in q_data_columns else stats.loc['Number of examinees']['Value']
        pdf.ln(ch / 2)
        if pdf.get_y() > 250:
            pdf.add_page()
        pdf.set_bg('heading_1')
        pdf.cell(w=pw, h=ch, txt=f"Question - {question}", ln=1, align='C', fill=True, border=1)
        pdf.set_bg('heading_2')
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
                    value = items.loc[
                        (items['title'] == question) & (items['answer'] == answer), col].values[0]
                    if col == 'answer':
                        txt = to_letter(value)
                    elif col == 'correct':
                        txt = '*' if value == 1 else ''
                    elif col == 'ticked':
                        txt = str(round(value)) + ' (' + str(
                            round((value / nb_presented * 100), 2)) + '%)'
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
                    pdf.cell(w=cols_3, h=ch, txt=txt, ln=0, align='C', fill=False, border=1)
        pdf.ln(ch)

    if pdf.get_y() > 100:
        pdf.add_page()
    pdf.set_font('helvetica', 'B', 16)
    pdf.start_section("Definitions")
    for key in definitions:
        pdf.ln(ch)
        pdf.set_font('helvetica', 'B', 14)
        pdf.start_section(key, level=1)
        # pdf.cell(w=pw, h=ch, txt=key, ln=1, align='L', fill=False, border=0)
        pdf.set_font('helvetica', '', 12)
        pdf.multi_cell(w=pw, h=ch, txt=definitions[key], align='L', fill=False, border=0)

    pdf.output(report_file_path, 'F')
    return report_file_path


def plot_charts(params):
    marks = params['marks']
    items = params['items']
    questions = params['questions']
    stats = params['stats']
    threshold = params['threshold']
    path = params['project_path']

    question_data_columns = ['presented', 'cancelled', 'replied', 'correct', 'empty', 'error', ]
    actual_data_columns = list(set(question_data_columns).intersection(questions.columns))
    question_analysis_columns = ['difficulty', 'discrimination', 'correlation', ]
    actual_analysis_columns = list(set(question_analysis_columns).intersection(questions.columns))

    outcome_data_columns = ['answer', 'correct', 'ticked', 'discrimination', ]
    actual_outcome_columns = list(set(outcome_data_columns).intersection(items.columns))

    image_path = path + '/img'
    # Create the directory
    os.makedirs(image_path, exist_ok=True)
    # Calculate the number of bins based on the maximum and minimum marks
    mark_bins = int(float(stats.loc['Maximum achieved mark', 'Value'])
                    - float(stats.loc['Minimum achieved mark', 'Value']))

    # create a histogram of the 'mark' column
    plt.subplots(1, 1, figsize=(9, 4))
    sns.histplot(marks['mark'], kde=True, bins=mark_bins)
    # Calculate the average value
    average_value = marks['mark'].mean()

    # Add a vertical line for the average value
    plt.axvline(average_value, color='red', linestyle='--',
                label=f'Mean ({round(average_value, 2)})')
    plt.xlabel('Mark')
    plt.ylabel('Number of students')
    plt.legend()
    # ax.set_title('Frequency of Marks')
    plt.savefig(image_path + '/marks.png', transparent=False, facecolor='white',
                bbox_inches="tight")

    # create a histogram of the 'mark' column
    diff_plot, ax2 = plt.subplots(1, 1, figsize=(9, 4))
    sns.histplot(questions['difficulty'], bins=30, color='blue')
    average_value = questions['difficulty'].mean()
    ax2.axvline(average_value, color='red', linestyle='--',
                label=f'Average ({round(average_value, 2)})')
    ax2.set_xlabel('Difficulty level (higher is easier)')
    ax2.set_ylabel('Number of questions')
    ax2.legend()
    # Set the color of the bars in the first histogram
    for patch in ax2.patches[:13]:
        patch.set_color('tab:red')
    for patch in ax2.patches[13:23]:
        patch.set_color('tab:blue')
    for patch in ax2.patches[23:]:
        patch.set_color('tab:green')
    plt.savefig(image_path + '/difficulty.png', transparent=False, facecolor='white',
                bbox_inches="tight")

    # create a histogram of discrimination if enough students
    if stats.loc['Number of examinees'][0] > threshold:
        fig, ax = plt.subplots(figsize=(9, 4))  # Set the figure size if desired
        sns.histplot(questions['discrimination'], bins=30, ax=ax, color='orange')
        average_value = questions['discrimination'].mean()
        ax.axvline(average_value, color='red', linestyle='--',
                   label=f'Average ({round(average_value, 2)})')
        ax.set_xlabel('Discrimination index (the higher the better)')
        ax.set_ylabel('Number of questions')
        ax.legend()
        # Set the color of the bars in the second histogram
        for patch in ax.patches[:13]:
            patch.set_color('tab:orange')
        for patch in ax.patches[13:23]:
            patch.set_color('tab:blue')
        for patch in ax.patches[23:]:
            patch.set_color('tab:green')
        plt.savefig(image_path + '/discrimination.png', transparent=False, facecolor='white',
                    bbox_inches="tight")

        # Plot difficulty vs discrimination
        fig, ax = plt.subplots(figsize=(9, 4))  # Set the figure size if desired
        sns.scatterplot(x=questions['discrimination'], y=questions['difficulty'], ax=ax)
        # Calculate the average values
        average_x = questions['discrimination'].mean()
        average_y = questions['difficulty'].mean()

        # Add vertical and horizontal lines for the average values
        ax.axhline(average_y, color='red', linestyle='--',
                   label=f'Average Difficulty ({round(average_y, 2)})')
        ax.axvline(average_x, color='blue', linestyle='--',
                   label=f'Average Discrimination ({round(average_x, 2)})')

        ax.set_xlabel('Discrimination index (the higher the better)')
        ax.set_ylabel('Difficulty level (higher is easier)')
        ax.legend()
        plt.savefig(image_path + '/discrimination_vs_difficulty.png', transparent=False,
                    facecolor='white',
                    bbox_inches="tight")

    # create a histogram of question correlation
    itemcorr_plot, ax3 = plt.subplots(1, 1, figsize=(9, 4))
    sns.histplot(questions['correlation'], kde=True, bins=mark_bins * 2)
    average_value = questions['correlation'].mean()
    ax3.axvline(average_value, color='red', linestyle='--',
                label=f'Average ({round(average_value, 2)})')
    ax3.set_xlabel('Item correlation')
    ax3.set_ylabel('Number of questions')
    ax3.legend()
    plt.savefig(image_path + '/item_correlation.png', transparent=False, facecolor='white',
                bbox_inches="tight")

    # create a bar chart for questions data columns
    # Get the values for the specified columns
    values = questions[actual_data_columns].mean()
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    sns.barplot(x=values, y=actual_data_columns, ax=ax)
    ax.set_xlabel('Average Number of Students')
    # Show the total number of questions on each bar
    for i, v in enumerate(values):
        ax.text(v + 3, i, str(v), color='black')

    # Save the plot
    plt.savefig(image_path + '/question_columns.png', transparent=False, facecolor='white',
                bbox_inches="tight")


if __name__ == '__main__':
    print("This package is note intended to be used standalone.")

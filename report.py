import datetime
import os

import pandas as pd
from fpdf import FPDF, TitleStyle
from fpdf.fonts import FontFace

today = datetime.datetime.now().strftime('%d/%m/%Y')


# ============================================================================
# CLASSIFICATION THRESHOLDS
# ============================================================================
# These thresholds are based on Classical Test Theory standards for
# classifying psychometric properties of exam questions

# === Difficulty Thresholds ===
DIFFICULTY_DIFFICULT_MAX = 0.4  # Questions with ≤40% correct are difficult
DIFFICULTY_INTERMEDIATE_MAX = 0.6  # Questions with 40-60% correct are intermediate
# Above 60% is considered easy

# === Discrimination Thresholds ===
DISCRIMINATION_LOW_MAX = 0.16  # Discrimination ≤0.16 is low
DISCRIMINATION_MODERATE_MAX = 0.3  # Discrimination 0.16-0.30 is moderate
DISCRIMINATION_HIGH_MAX = 0.5  # Discrimination 0.30-0.50 is high
# Above 0.50 is very high, negative values need review

# === Correlation Thresholds ===
CORRELATION_NONE_MAX = 0.1  # Point-biserial correlation ≤0.1 is negligible
CORRELATION_LOW_MAX = 0.2  # Correlation 0.1-0.2 is low
CORRELATION_MODERATE_MAX = 0.3  # Correlation 0.2-0.3 is moderate
CORRELATION_STRONG_MAX = 0.45  # Correlation 0.3-0.45 is strong
# Above 0.45 is very strong, negative values need review


def to_letter(value: int) -> str:
    """
    Convert a value to a letter
    :param value: value to convert (int)
    :return: letter corresponding to the value (str)
    """
    letter: str = chr(ord('A') + value - 1)
    return letter


class PDF(FPDF):
    def __init__(self, project_name, colour_palette, name, url):
        super().__init__()
        # Margin
        self.margin: int = 10
        # Page width: Width of A4 is 210mm
        self.pw: int = 210 - 2 * self.margin
        # Cell height
        self.ch: int = 6
        self.project: str = project_name
        self.colour_palette: dict = colour_palette
        self.name: str = name
        self.url: str = url

    def header(self):
        self.set_font('Helvetica', '', 10)
        self.cell(w=self.epw / 2, h=9, txt=f'{self.project} - Exam report', border=0, ln=0,
                  align='L')
        self.cell(w=self.epw / 2, h=9, txt=f"{today}", ln=1, align='R')

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', '', 10)
        self.cell(w=self.epw / 3, h=8,
                  txt=f"©{datetime.datetime.now().strftime('%Y')} - {self.name}",
                  border=0, ln=0, align='L')
        self.cell(self.epw / 3, 8, f'Page {self.page_no()}', 0, 0, 'C')
        self.cell(w=self.epw / 3, h=8, txt=self.url, border=0, ln=0, align='R')

    def set_bg(self, colour):
        self.set_fill_color(self.colour_palette[colour][0],
                            self.colour_palette[colour][1],
                            self.colour_palette[colour][2])
        self.set_text_color(self.colour_palette[colour][3])


def p(pdf, text, **kwargs):
    """Inserts a paragraph"""
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


def rnd_float(df, digits):
    """
    Round float type values of a dataframe to the number of digits in args
    :param df: pandas dataframe
    :param digits: number of digits for rounding
    :type df:
    :return: df
    :rtype: pandas dataframe
    """
    rnd_format = "{:." + str(digits) + "f}"
    for col in df.columns:
        df[col] = df[col].apply(lambda x: rnd_format.format(x) if isinstance(x, float) else x)
    return df


def render_table(df, pdf):
    # Set columns with float values as 4 digits with trailing zeros
    df = rnd_float(df, 4)
    df = df.map(str)  # Convert all data inside dataframe into string type

    columns = [list(df)]  # Get list of dataframe columns
    rows = df.values.tolist()  # Get list of dataframe rows
    # Define column alignment: Centered for title, right for other columns
    text_align = ['C' if col == 'title' else 'R' for col in columns[0]]
    data = columns + rows  # Combine columns and rows in one list
    bg = pdf.colour_palette['heading_2'][:3]
    fg = pdf.colour_palette['heading_2'][3]
    headings_style = FontFace(emphasis="BOLD", color=fg, fill_color=bg)
    with pdf.table(cell_fill_color=200,  # grey
                   cell_fill_mode="ROWS",  # Doesn't seem to work
                   headings_style=headings_style,
                   text_align=text_align,
                   width=pdf.pw) as table:
        for data_row in data:
            row = table.row()
            for datum in data_row:
                row.cell(datum)


def render_headers(df, pdf, cw):
    """
    Prints the headers of a table. This migh be called several times for the same
    data as there may be new pages in the middle of the table.
    :param cw: Column width
    :type cw:
    :param df: the dataframe to print headers from
    :type df:
    :param pdf:
    :type pdf:
    :return: nothing
    :rtype:
    """
    for col in df.columns:
        pdf.cell(w=cw, h=pdf.ch, txt=f"{col}", ln=0, align='C', fill=True, border=1)


def get_label(col: str, value: float) -> tuple:
    """
    Define the labels to apply to a value for difficulty, discrimination and correlation
    :param col: Name of the column to evaluate
    :type col: str
    :param value: value of the index to evaluate
    :type value: int or float
    :return: label, fill_color
    :rtype: str
    """
    if col == 'difficulty':
        return get_difficulty_label(value)
    elif col == 'discrimination':
        return get_discrimination_label(value)
    elif col == 'correlation':
        return get_correlation_label(value)
    else:
        return '-', 'white'


def get_difficulty_label(value: float) -> tuple:
    """
    Classify difficulty level based on percentage of correct answers.

    Args:
        value: Difficulty index (0-1, where higher = easier)

    Returns:
        Tuple of (label, color) for display
    """
    if value <= DIFFICULTY_DIFFICULT_MAX:
        return 'Difficult', 'red'
    elif value <= DIFFICULTY_INTERMEDIATE_MAX:
        return 'Intermediate', 'yellow'
    else:
        return 'Easy', 'green'


def get_discrimination_label(value: float) -> tuple:
    """
    Classify discrimination index based on CTT standards.

    Args:
        value: Discrimination index (typically -1 to 1)

    Returns:
        Tuple of (label, color) for display
    """
    if value < 0:
        return 'Review!', 'red'
    elif value <= DISCRIMINATION_LOW_MAX:
        return 'Low', 'grey'
    elif value <= DISCRIMINATION_MODERATE_MAX:
        return 'Moderate', 'yellow'
    elif value <= DISCRIMINATION_HIGH_MAX:
        return 'High', 'green'
    else:
        return 'Very high', 'blue'


def get_correlation_label(value: float) -> tuple:
    """
    Classify point-biserial correlation based on CTT standards.

    Args:
        value: Point-biserial correlation coefficient (-1 to 1)

    Returns:
        Tuple of (label, color) for display
    """
    if value < 0:
        return 'Review!', 'red'
    elif value <= CORRELATION_NONE_MAX:
        return 'None', 'white'
    elif value <= CORRELATION_LOW_MAX:
        return 'Low', 'grey'
    elif value <= CORRELATION_MODERATE_MAX:
        return 'Moderate', 'yellow'
    elif value <= CORRELATION_STRONG_MAX:
        return 'Strong', 'green'
    else:
        return 'Very strong', 'blue'


def generate_pdf_report(params: dict):
    """
    https://towardsdatascience.com/how-to-create-a-pdf-report-for-your-data-analysis-in-python-2bea81133b
    The ln parameter defines where the position should go after this cell:
    0: to the right of the current cell
    1: to the beginning of the next line
    2: below the current cell
    :return:
    """
    project_name: str = params['project_name']
    stats: pd.DataFrame = params['stats']
    questions: pd.DataFrame = params['questions']
    items: pd.DataFrame = params['items']
    threshold: int = params['threshold']
    definitions: dict = params['definitions']
    findings: dict = params['findings']
    blurb: str = params['blurb']
    palette: dict[str, tuple[int, int, int, int]] = params['palette']
    report_path: str = params['project_path']
    image_path: str = report_path + '/img'
    report_file_path: str = report_path + '/' + project_name + '-report.pdf'
    company: str = params['company_name']
    url: str = params['company_url']
    correction: str = params['correction']

    question_data_columns: list = ['presented', 'cancelled', 'replied', 'correct', 'empty', 'error', ]
    actual_data_columns: list = [col for col in question_data_columns if col in questions.columns]
    question_analysis_columns: list = ['difficulty', 'discrimination', 'correlation', ]
    actual_analysis_columns: list = [col for col in question_analysis_columns if col in questions.columns]

    outcome_data_columns: list = ['answer', 'correct', 'ticked', 'discrimination', ]

    pdf: PDF = PDF(project_name, palette, company, url)
    ch: int = pdf.ch
    pw: int = pdf.pw
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(script_dir, 'logo.png')
    if not os.path.isfile(logo_path):
        pdf.cell(w=pw, h=ch, txt="Your logo goes here. Place a 'logo.png' in the same folder",
                 border=0, ln=1, align='C')
    pdf.image(logo_path, x=pw / 2 - 10, y=None, w=40, h=0, type='PNG')
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
        # if key in ['Number of examinees', 'Number of questions'], make it integer with no decimal
        if key in ['Number of examinees', 'Number of questions']:
            text = str(round(stats[key]))
        else:
            text = str(stats[key])
        pdf.cell(w=pw / 5, h=ch, txt=text, ln=0, align='C', border='LBR')
    pdf.ln(ch)
    # Second row of data
    pdf.set_font('Helvetica', 'B', 12)
    for text in ['Mean', 'Median', 'Mode', 'Std Dev', 'Variance']:
        pdf.cell(w=pw / 5, h=6, txt=text, ln=0, align='C', fill=True, border=1)
    pdf.ln(ch)
    pdf.set_font('Helvetica', '', 12)
    for key in ['Mean', 'Median', 'Mode', 'Standard deviation', 'Variance']:
        text = str(round(stats[key], 2))
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
    stats['Average Difficulty'] = questions['difficulty'].mean()
    pdf.set_font('Helvetica', '', 12)
    for key in ['Standard error of mean', 'Standard error of measurement', 'Skew', 'Kurtosis',
                'Average Difficulty']:
        text = str(round(stats[key], 2))
        pdf.cell(w=pw / 5, h=ch, txt=text, ln=0, align='C', border='LBR')
    pdf.ln(ch + 3)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(w=pw / 2, h=6, txt="Frequency of marks", ln=0, align='C', fill=True, border=1)
    x = pdf.get_x()
    pdf.cell(w=pw / 2, h=6, txt="Difficulty levels", ln=1, align='C', fill=True, border=1)
    y = pdf.get_y()
    pdf.image(image_path + '/marks.png', w=pw / 2, type='PNG')
    pdf.image(image_path + '/difficulty.png', w=pw / 2, x=x, y=y, type='PNG')
    if stats['Number of examinees'] > threshold:
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
    pdf.set_font('Helvetica', '', 12)
    pdf.multi_cell(w=pw, h=ch, txt=correction, markdown=True)
    pdf.add_page()
    # Display the overall findings
    pdf.start_section("Findings")
    pdf.start_section("Summary (TL;DR)", level=1)
    pdf.set_font('Helvetica', '', 12)
    pdf.multi_cell(w=pw, h=ch, txt=blurb, markdown=True)
    pdf.set_bg('white')
    # Display details of findings
    for key in findings.keys():
        column = findings[key]['column']
        limit = findings[key]['limit']
        comparison = findings[key]['comparison_operator']
        # Create the condition dynamically using the comparison operator
        if column != 'cancelled':
            condition = f"questions['{column}'] {comparison} {limit}"
        else:
            condition = f"questions['{column}'] {comparison} questions['presented'] * {limit}"
            limit = 80
        if column in questions.columns and questions[eval(condition)].shape[0] > 0:
            data = questions[eval(condition)]
            if column != 'cancelled':
                data = data[['title'] + actual_analysis_columns] \
                    .sort_values(by=column, ascending=True if comparison == '<' else False)
            else:
                data = data[['title'] + actual_data_columns + ['difficulty'] + ['correlation']] \
                    .sort_values(by=column, ascending=True if comparison == '<' else False)
                data.drop('empty', axis=1, inplace=True)
            heading = findings[key]['heading']
            text = findings[key]['text']
            nb_questions = data.shape[0]
            plural = 's have' if nb_questions > 1 else ' has'
            txt = text.format(questions=nb_questions,
                              total=questions.shape[0],
                              plural=plural,
                              limit=limit,
                              percent=str(round(
                                  100 * nb_questions / questions.shape[0], 2)))
            pdf.start_section(f"{heading}", level=1)
            pdf.multi_cell(w=pw, txt=txt, markdown=True, ln=1)
            render_table(data, pdf)

    q_data_columns = []
    q_analysis_columns = []
    o_data_columns = []
    all_q_columns = ['title'] + actual_data_columns + actual_analysis_columns
    for col in question_data_columns:
        if (col in questions.columns) and (questions[col].std() != 0):
            q_data_columns.append(col)
    for col in question_analysis_columns:
        if (col in questions.columns) and (questions[col].std() != 0):
            q_analysis_columns.append(col)
    for col in outcome_data_columns:
        if (col in items.columns) and (items[col].std() != 0):
            o_data_columns.append(col)
    # Display the condensed summary of questions
    overview = questions[all_q_columns].sort_values(by='title', ascending=True).reset_index(drop=True)
    overview = rnd_float(overview, 4)
    pdf.add_page(orientation='landscape', format='a4')
    pdf.ln(ch * 0.75)
    pdf.start_section("Items overview", level=0)
    cw = round(pdf.epw / len(all_q_columns))
    w = pdf.epw / 13
    title_w = pdf.epw / 10
    pdf.set_font('Helvetica', 'B', 12)
    # pdf.cell(w=title_w, h=ch, txt='Legend:', ln=1, align='L', fill=False, border=0)
    # Display the legend for colours
    for col in q_analysis_columns:
        pdf.set_font('Helvetica', 'B', 10)
        if col == 'difficulty':
            pdf.cell(w=title_w, h=ch, txt='Difficulty:', ln=0, align='L', fill=False, border=0)
            # pdf.cell(w=2, h=ch, txt=f"", ln=0, align='L', fill=False, border=0)
            pdf.set_font('Helvetica', '', 10)
            for level, color in zip(['Difficult', 'Intermediate', 'Easy'], ['red', 'yellow', 'green']):
                pdf.set_bg(color)
                pdf.cell(w=w, h=ch, txt=f"{level}", ln=0, align='C', fill=True, border=1)
                # pdf.cell(w=2, h=ch, txt=f"", ln=0, align='L', fill=False, border=0)
        elif col == 'discrimination':
            pdf.cell(w=title_w, h=ch, txt='', ln=0, align='L', fill=False, border=0)
            pdf.cell(w=title_w, h=ch, txt='Discrimination:', ln=0, align='L', fill=False, border=0)
            pdf.set_font('Helvetica', '', 10)
            for level, color in zip(['Review!', 'Low', 'Moderate', 'High', 'Very high'],
                                    ['red', 'grey', 'yellow', 'green', 'blue']):
                pdf.set_bg(color)
                pdf.cell(w=w, h=ch, txt=f"{level}", ln=0, align='C', fill=True, border=1)
                # pdf.cell(w=2, h=ch, txt=f"", ln=0, align='L', fill=False, border=0)
        elif col == 'correlation':
            pdf.ln(ch * 1.25)
            pdf.cell(w=title_w, h=ch, txt='Correlation:', ln=0, align='L', fill=False, border=0)
            pdf.set_font('Helvetica', '', 10)
            for level, color in zip(['Review!', 'None', 'Low', 'Moderate', 'Strong', 'Very strong'],
                                    ['red', 'grey', 'white', 'yellow', 'green', 'blue']):
                pdf.set_bg(color)
                pdf.cell(w=w, h=ch, txt=f"{level}", ln=0, align='C', fill=True, border=1)
                # pdf.cell(w=2, h=ch, txt=f"", ln=0, align='L', fill=False, border=0)
    pdf.ln(ch * 1.25)
    pdf.set_bg('heading_2')
    pdf.set_font('Helvetica', 'B', 12)
    render_headers(overview, pdf, cw)
    pdf.set_font('Helvetica', '', 10)
    for index, row in overview.iterrows():
        # Iterate through each row in the dataframe
        pdf.ln(ch)
        # Print headers again on a new page
        if (pdf.get_y() > pdf.eph + ch * 0.75) or (pdf.get_y() < pdf.margin * 2 + 10):
            pdf.set_font('Helvetica', 'B', 12)
            pdf.set_bg('heading_2')
            render_headers(overview, pdf, cw)
            pdf.set_font('Helvetica', '', 10)
            pdf.set_bg('white')
            pdf.ln(ch)
        for column in overview.columns:
            # Iterate through each column in the dataframe
            value = row[column]
            label, fill_color = get_label(column, float(value)) if column != 'title' \
                else ('', 'white')
            align = 'R' if column != 'title' else 'C'
            pdf.set_bg(fill_color)
            pdf.cell(w=cw, h=ch, txt=f"{value}", ln=0, align=align, fill=True, border=1)

    # Display the details of the question data
    pdf.add_page()
    pdf.start_section("Items and Outcomes (detailed)", level=0)
    cols_1 = pw / len(q_data_columns)  # Presented, cancelled...
    cols_2 = pw / len(q_analysis_columns)  # Difficulty, Discrimination...
    cols_3 = pw / len(o_data_columns)  # Answer, Correct, Ticked...
    p(pdf, text="This section presents all the items and outcomes of the examination in detail.\
             \nSome values are colour coded for clarity. The colour code is as follows:")
    for col in q_analysis_columns:
        pdf.set_font('Helvetica', 'B', 12)
        pdf.ln(ch / 2)
        if col == 'difficulty':
            pdf.cell(w=pw, h=ch, txt='Difficulty:', ln=1, align='L', fill=False, border=0)
            pdf.set_font('Helvetica', '', 12)
            for level, color in zip(['Difficult', 'Intermediate', 'Easy'], ['red', 'yellow', 'green']):
                pdf.set_bg(color)
                pdf.cell(w=pw / 7, h=ch, txt=f"{level}", ln=0, align='C', fill=True, border=1)
                pdf.cell(w=2, h=ch, txt="", ln=0, align='L', fill=False, border=0)
            pdf.ln(ch)
        elif col == 'discrimination':
            pdf.cell(w=pw, h=ch, txt='Discrimination:', ln=1, align='L', fill=False, border=0)
            pdf.set_font('Helvetica', '', 12)
            for level, color in zip(['Review!', 'Low', 'Moderate', 'High', 'Very high'],
                                    ['red', 'grey', 'yellow', 'green', 'blue']):
                pdf.set_bg(color)
                pdf.cell(w=pw / 7, h=ch, txt=f"{level}", ln=0, align='C', fill=True, border=1)
                pdf.cell(w=2, h=ch, txt="", ln=0, align='L', fill=False, border=0)
            pdf.ln(ch)
        elif col == 'correlation':
            pdf.cell(w=pw, h=ch, txt='Correlation:', ln=1, align='L', fill=False, border=0)
            pdf.set_font('Helvetica', '', 12)
            for level, color in zip(['Review!', 'None', 'Low', 'Moderate', 'Strong', 'Very strong'],
                                    ['red', 'grey', 'white', 'yellow', 'green', 'blue']):
                pdf.set_bg(color)
                pdf.cell(w=pw / 7, h=ch, txt=f"{level}", ln=0, align='C', fill=True, border=1)
                pdf.cell(w=2, h=ch, txt="", ln=0, align='L', fill=False, border=0)
            pdf.ln(ch)

    for question in questions.sort_values('title')['title'].values:
        nb_presented = questions[questions['title'] == question]['presented'].values[0] \
            if 'presented' in q_data_columns else stats['Number of examinees']
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
            label, fill_color = get_label(col, value)
            txt = f"{round(value, 4)} ({label})"
            pdf.set_bg(fill_color)
            pdf.cell(w=cols_2, h=ch, txt=txt, ln=0, align='C', fill=True, border=1)
        pdf.ln(ch)
        # Breakdown of question's outcomes
        if items.loc[(items['title'] == question), 'answer'].count() > 10:
            continue
        for col in o_data_columns:
            pdf.set_bg('heading_2')
            pdf.cell(w=cols_3, h=ch, txt=f"{col}", ln=0, align='C', fill=True, border=1)
        for answer in items.loc[items['title'] == question, 'answer'].values:
            pdf.ln(ch)
            for col in o_data_columns:
                pdf.set_bg('white')
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
                    else:
                        label, fill_color = get_label(col, value)
                        pdf.set_bg(fill_color)
                    txt = label if label == '-' else str(round(value, 4)) + ' (' + label + ')'
                pdf.cell(w=cols_3, h=ch, txt=txt, ln=0, align='C', fill=True, border=1)
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


if __name__ == '__main__':
    print("This package is note intended to be used standalone.")

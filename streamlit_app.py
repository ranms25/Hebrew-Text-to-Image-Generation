import streamlit as st
from generate_and_enhanced_best_sentences import (
    generate_best_sentence,
    generate_enhanced_text,
    extract_main_characters,
    generate_general_descriptions
)
from google_translator import translate_text, align_hebrew_best_sentences
from paragraphs_summary import paragraph_summary
from text_segmentation import segment_text
from text_to_image import get_image_from_primary, get_image_from_space
import base64
from streamlit_js_eval import streamlit_js_eval
import platform
import pdfkit
import time
import asyncio
import aiohttp
from PIL import Image
import io
from gradio_client import Client
import requests

API_URL_CORE = "https://ranm-coreferenceresolution.hf.space/predict"

# Load the CSS file for custom styling
with open('style.css', encoding='utf-8') as f:
    css = f.read()

# Apply CSS styles to the Streamlit app
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


# Determine the path to wkhtmltopdf based on the operating system
def get_wkhtmltopdf_path():
    """
    Returns the path to the wkhtmltopdf binary based on the operating system.
    This is needed to convert HTML to PDF using PDFKit.
    """
    if platform.system() == 'Windows':
        # Path on Windows - adjust the path to where wkhtmltopdf is installed on your local machine
        return r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    else:
        # Path on Unix/Linux - typical location on Ubuntu
        return '/usr/bin/wkhtmltopdf'


# Set up PDFKit configuration with the correct path
config = pdfkit.configuration(wkhtmltopdf=get_wkhtmltopdf_path())


def toggle_page(content="", show=True):
    """
    Function to toggle the visibility of an overlay on the page with customizable content.
    This is used to provide feedback to the user during processing, such as showing a loading message.

    Parameters:
    - content: The message to be displayed on the overlay.
    - show: A boolean flag to show or hide the overlay.
    """
    if show:
        code = f"""
        <style>
          body::before {{
            content: '{content}';
            position: fixed;
            left: 0;
            top: 0;
            right: 0;
            bottom: 0;
            background: white;
            z-index: 9999;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 40px;
            font-family: 'Calibri';
          }}
        </style>
        """
    else:
        code = """
        <style>
          body::before {
            display: none;
          }
        </style>
        """
    st.markdown(code, unsafe_allow_html=True)


def on_done_button_click(hebrew_text, hebrew_best_sentences_each_par, title_of_story, author_name):
    """
    Function to handle the 'Done' button click, process the selected images,
    and generate a downloadable PDF with the story and images.

    Parameters:
    - hebrew_text: The original Hebrew text of the story.
    - hebrew_best_sentences_each_par: List of tuples containing the best sentences in Hebrew for each paragraph.
    - title_of_story: The title of the story.
    - author_name: The name of the author.
    """
    toggle_page(content="🎨 יוצר את הסיפור")
    st.session_state.done = True  # Set the "done" state to True

    # Ensure that there are selected images or proceed without images
    if len(st.session_state.selected_images) == 0 or all(image_info[0] is None for image_info in st.session_state.selected_images):
        return

    # Ensure `selected_images` list is consistent with paragraphs
    consistent_images = []
    for i, image_info in enumerate(st.session_state.selected_images):
        if st.session_state.get(f"checkbox_{i}"):
            consistent_images.append(image_info)
        else:
            consistent_images.append(None)

    # Update session state with the consistent list of images
    st.session_state.selected_images = consistent_images

    # Create a list to store the content (text + images) for the PDF
    downloadable_content = []
    index_start_hebrew_text = 0

    # Use an expander to wrap the results
    with st.expander("", expanded=True):
        for i, image_info in enumerate(st.session_state.selected_images):
            # Get the sentence from the paragraph mapping
            sentence = hebrew_best_sentences_each_par[i][0]

            # Find the correct position of the sentence in the Hebrew text
            end_index = hebrew_text.find(sentence, index_start_hebrew_text)
            if end_index == -1:
                # If the sentence is not found, continue with the next paragraph
                continue

            # Ensure we capture the sentence including the period (.) at the end
            end_index += len(sentence)

            # Check if the next character is a period or punctuation, and include it
            if end_index < len(hebrew_text) and hebrew_text[end_index] in ['.', '!', '?']:
                end_index += 1  # Include the punctuation in the current paragraph

            # Extract and display the paragraph text (only once)
            paragraph_text = hebrew_text[index_start_hebrew_text:end_index]
            downloadable_content.append(paragraph_text)
            st.markdown(f'<p class="hebrew-text">{paragraph_text}</p>', unsafe_allow_html=True)

            # Update the start index for the next paragraph
            index_start_hebrew_text = end_index

            # Add the image if it exists and was selected
            if image_info is not None:
                image_bytes, _ = image_info  # `_` because we don't need to check `selected` anymore
                # Display the image below the corresponding paragraph
                st.image(image_bytes)
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                downloadable_content.append(image_base64)  # Add image to the content list
            else:
                # Add a placeholder or just skip appending an image to maintain content alignment
                downloadable_content.append(None)

        # Append any remaining text after the last paragraph
        if index_start_hebrew_text < len(hebrew_text):
            remaining_text = hebrew_text[index_start_hebrew_text:]
            downloadable_content.append(remaining_text)
            st.markdown(f'<p class="hebrew-text">{remaining_text}</p>', unsafe_allow_html=True)

        # Reset session state flags
        st.session_state.done_button_clicked = False
        st.session_state.selected_images = []

        # Create the HTML content for the PDF
        html_content = create_html_content(downloadable_content, title_of_story, author_name)

        # Generate the PDF from the HTML content
        pdf_result = generate_pdf(html_content)

        # Add a download button for the generated PDF
        download_label = "לחצו כאן להורדת הסיפור שלכם עם האיורים"
        download_button_html = f"""
        <a href="data:application/pdf;base64,{base64.b64encode(pdf_result).decode()}" download="{title_of_story}.pdf">
            <div class="download-button">{download_label}</div>
        </a>
        """
        st.markdown(download_button_html, unsafe_allow_html=True)

        # Create a second PDF with only the images and add a download button
        download_images_label = "לחצו כאן להורדת כל האיורים"
        download_images_content = create_html_content(downloadable_content, title_of_story, author_name,
                                                      only_images=True)
        convert_to_pdf = generate_pdf(download_images_content)
        download_images_button_html = f"""
        <a href="data:application/pdf;base64,{base64.b64encode(convert_to_pdf).decode()}" download="{title_of_story}_images.pdf">
            <div class="download-button">{download_images_label}</div>
        </a>
        """
        st.markdown(download_images_button_html, unsafe_allow_html=True)

        # Hide the loading overlay after processing
        time.sleep(6)
        toggle_page(show=False)


def create_html_content(content, title_of_story, author_name, only_images=False):
    """
    Function to create the HTML content that will be converted to a PDF.
    The content includes the story title, author name, text, and images.

    Parameters:
    - content: List containing text and images in alternating order.
    - title_of_story: The title of the story.
    - author_name: The name of the author.
    - only_images: A boolean flag indicating whether to generate content with only images (True) or both text and images (False).

    Returns:
    - html_content: A string containing the HTML structure.
    """
    html_content = f'''
    <html>
    <head>
    <style>
    {css}
    '''

    if only_images:
        # Apply margin only if generating the images-only PDF
        html_content += '''
        .image-spacing {
            margin-bottom: 20px;  /* Adjust the margin value as needed */
            display: block;
            width: 100%;  /* Make images full width */
        }
        '''

    html_content += '''
    </style>
    </head>
    <body class="pdf-body">
    <div dir="rtl">
    '''
    if not only_images:
        html_content += f'<h2 class="custom-title-pdf">{title_of_story}</h2>'
        html_content += f'<h3 class="custom-author-pdf">{author_name}</h3>'

    for i, item in enumerate(content):
        if i % 2 == 0:  # Even indices contain text
            if not only_images:
                html_content += f'<p class="hebrew-text-pdf">{item}</p>'
        else:  # Odd indices contain images
            if item:  # Only add the <img> tag if there is an image
                if only_images:
                    img_tag = f'<div class="image-spacing"><img src="data:image/png;base64,{item}" alt="Image {i // 2}" class="resized-image-pdf"></div>'
                else:
                    img_tag = f'<img src="data:image/png;base64,{item}" alt="Image {i // 2}" class="resized-image-pdf">'
                html_content += f'{img_tag}'

    html_content += '</div></body></html>'
    return html_content


def generate_pdf(html_content):
    """
    Function to convert HTML content into a PDF file using PDFKit.

    Parameters:
    - html_content: The HTML content as a string.

    Returns:
    - pdf: The binary content of the generated PDF.
    """
    options = {
        'page-size': 'A4',
        'encoding': "UTF-8",
        'custom-header': [('Accept-Encoding', 'gzip')],
        'no-outline': None,
        'enable-local-file-access': None,
        'zoom': '1.25',
        'margin-top': '20mm',
        'margin-right': '15mm',
        'margin-bottom': '20mm',
        'margin-left': '15mm',
        'header-left': '[title]',
        'header-font-size': '14',
        'footer-center': '[page]',
        'footer-font-size': '14',
    }
    pdf = pdfkit.from_string(html_content, False, options=options, configuration=config)
    return pdf


def show_alert(message="", show=True):
    """
    Function to display or hide an alert overlay on the page.
    The alert includes a message and a spinner icon.

    Parameters:
    - message: The message to display in the alert.
    - show: A boolean flag to show or hide the alert.
    """
    if show:
        code = f""" <div id="alert" style="background-color: rgba(255, 255, 255, 0.75); position: fixed; top: 0; 
        left: 0; width: 100%; height: 100%; z-index: 9998;"> <div style="background-color: white; padding: 20px; 
        position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); border-radius: 5px; border: 2px 
        solid #6a00ff; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); font-size: 20px; font-family: 'Calibri'; display: 
        flex; align-items: center; justify-content: center;"> <div style="text-align: center;">{message}</div> <div 
        id="spinner" style="border: 4px solid rgba(0, 0, 0, 0.1); border-top: 4px solid #6a00ff; border-radius: 50%; 
        width: 30px; height: 30px; animation: spin 1s linear infinite; margin-left: 10px; order: -1;"></div> </div> 
        </div> <style> @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }} 
        </style> """
    else:
        code = """
        <style>
          #alert {
            display: none;
          }
        </style>
        """
    st.markdown(code, unsafe_allow_html=True)


async def fetch_images(primary_tasks, space_tasks, gradio_clients):
    """
    Asynchronously fetch images either from the primary model or fallback to the space model if the primary model fails.

    Parameters:
    - primary_tasks: A dictionary of primary tasks to fetch images from the main model.
    - space_tasks: A dictionary of tasks to fetch images from the fallback space model.
    - gradio_clients: A list of Gradio client objects for accessing the Hugging Face space models.

    Returns:
    - primary_results_dict: A dictionary of image bytes indexed by paragraph number.
    """
    primary_results = await asyncio.gather(*primary_tasks.values(), return_exceptions=True)
    invalid_images = {k: v for k, v in zip(primary_tasks.keys(), primary_results) if
                      isinstance(v, Exception) or not (v and v.startswith(b"\xff\xd8"))}
    primary_results_dict = {paragraph_number: result for paragraph_number, result in
                            zip(primary_tasks.keys(), primary_results)}

    if not primary_tasks:  # Check if primary_tasks is empty
        invalid_images = {k: None for k in space_tasks.keys()}  # All images need to be processed by space API

    if not invalid_images:
        return primary_results_dict

    show_alert(
        "אופס! שיבוש קל בתהליך, אם תרצו לקבל תוצאות טובות יותר נסו מאוחר יותר, או המתינו והתוצאות הנוכחיות יתקבלו תוך "
        "כ-5 דקות.",
        show=True)
    num_clients = len(gradio_clients)
    space_tasks_distributed = {k: (space_tasks[k][0], gradio_clients[i % num_clients], *space_tasks[k][1:]) for i, k in
                               enumerate(invalid_images.keys())}
    space_results = await asyncio.gather(
        *[space_tasks_distributed[k][0](space_tasks_distributed[k][1], *space_tasks_distributed[k][2:]) for k in
          invalid_images.keys()]
    )
    for k, v in zip(invalid_images.keys(), space_results):
        if v:
            primary_results_dict[k] = v

    show_alert("", show=False)
    return primary_results_dict


async def fetch_images_async(model_name, best_sentences, sentence_mapping, general_descriptions, selected_style,
                             hebrew_best_sentences_each_par):
    """
    Asynchronously fetch images generated by the models based on the best sentences from the story.

    Parameters:
    - model_name: The name of the primary model to generate images.
    - best_sentences: List of best sentences generated from the story.
    - sentence_mapping: Mapping of enhanced sentences to their original positions.
    - general_descriptions: General descriptions of the characters for additional context.
    - selected_style: The chosen style for the images.
    - hebrew_best_sentences_each_par: List of best sentences in Hebrew for each paragraph.

    The function distributes tasks to fetch images from both primary and fallback space models.
    """
    grouped_sentences = group_sentences_by_paragraph(best_sentences)
    # st.write(f'grouped_sentences:{grouped_sentences}')
    # st.write(f'hebrew_best_sentences_each_par:{hebrew_best_sentences_each_par}')
    gradio_clients = [
        Client("RanM/text2image_1"),
        Client("RanM/text2image_2"),
        Client("RanM/text2image_3"),
        Client("RanM/text2image_4"),
        Client("RanM/text2image_5")
    ]

    async with aiohttp.ClientSession() as session:
        primary_tasks = {}
        if model_name is not None:
            primary_tasks = {
                paragraph_number: get_image_from_primary(
                    session, model_name, " ".join(sentences), paragraph_number,
                    sentence_mapping, general_descriptions, selected_style, hebrew_best_sentences_each_par
                )
                for paragraph_number, sentences in grouped_sentences.items()
            }

        space_tasks = {
            paragraph_number: (get_image_from_space,
                               paragraph_number, {paragraph_number: sentences}, general_descriptions, selected_style)
            for paragraph_number, sentences in grouped_sentences.items()
        }

        primary_results = await fetch_images(primary_tasks, space_tasks, gradio_clients)

        for paragraph_number, image_bytes in primary_results.items():
            if image_bytes is not None:
                st.subheader(f"איור לפסקה  {paragraph_number + 1}")
                image = Image.open(io.BytesIO(image_bytes)).resize((512, 512))
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                caption = hebrew_best_sentences_each_par[paragraph_number][0]
                st.markdown(
                    f'<div style="width: 640px; margin-bottom: 20px; text-align: right;">'
                    f'<img src="data:image/png;base64,{image_base64}" style="width:640px;height:640px;" alt="Image {paragraph_number + 1}"> '
                    f'<div style="font-family: Calibri; font-size: 18px; direction: rtl; text-align: right; '
                    f'margin-top: 10px;">{caption}</div> '
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.session_state.selected_images.append(
                    (image_bytes,
                     st.checkbox(f" איור מספר {paragraph_number + 1}", key=f"checkbox_{paragraph_number}", value=True))
                )


def show_enhanced_images(model_name, best_sentences, sentence_mapping, general_descriptions, selected_style,
                         hebrew_best_sentences_each_par):
    """
    Fetch and display enhanced images in the Streamlit app. The images are generated based on the best sentences
    extracted from the story.

    Parameters:
    - model_name: The name of the primary model to generate images.
    - best_sentences: List of best sentences generated from the story.
    - sentence_mapping: Mapping of enhanced sentences to their original positions.
    - general_descriptions: General descriptions of the characters for additional context.
    - selected_style: The chosen style for the images.
    - hebrew_best_sentences_each_par: List of best sentences in Hebrew for each paragraph.
    """
    st.session_state.selected_images = []
    asyncio.run(fetch_images_async(model_name, best_sentences, sentence_mapping, general_descriptions, selected_style,
                                   hebrew_best_sentences_each_par))


def group_sentences_by_paragraph(sentences):
    """
    Groups the enhanced sentences by their respective paragraph numbers
    and ensures they are sorted in the correct order.

    Parameters:
    - sentences: List of enhanced sentences with paragraph numbers.

    Returns:
    - grouped_sentences: A dictionary where keys are paragraph numbers and
      values are lists of enhanced sentences, sorted by paragraph numbers.
    """
    grouped_sentences = {}
    for sentence_info in sentences:
        paragraph_number, enhanced_sentence, _ = sentence_info
        if paragraph_number not in grouped_sentences:
            grouped_sentences[paragraph_number] = []
        grouped_sentences[paragraph_number].append(enhanced_sentence)

    # Sort paragraphs by their paragraph number to ensure correct display order
    sorted_grouped_sentences = {k: grouped_sentences[k] for k in sorted(grouped_sentences.keys())}
    # st.write(f'sorted_grouped_sentences:{sorted_grouped_sentences}')
    return sorted_grouped_sentences


def check_model_availability(model_names):
    """
    Check the availability of the models on Hugging Face.

    Parameters:
    - model_names: List of model names to check.

    Returns:
    - The first available model name. Returns None if no models are available.
    """
    for model_name in model_names:
        # print(f"Checking model availability: {model_name}")
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {
            "Authorization": f"Bearer {st.secrets.hf_credentials.header}"
        }
        payload = {
            "inputs": "Checking if the model is loading"
        }
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            # print(f"Model {model_name} ok model name: {model_name}")
            return model_name
        else:
            error_message = response.json().get("error", "")
            # print(f"Model {model_name} error: {error_message}")
    return None


def restart():
    """
    Function to reload the Streamlit app, effectively resetting the session state.
    """
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


def main():
    """
    The main function to run the Streamlit app, handling user inputs, generating images, and creating the final PDF.
    """
    # Initialize session state variables
    if "done" not in st.session_state:
        st.session_state.done = False
    if "translate" not in st.session_state:
        st.session_state.translate = False
    if "selected_images" not in st.session_state:
        st.session_state.selected_images = []
    if "user_input_hebrew" not in st.session_state:
        st.session_state.user_input_hebrew = ""
    if "selected_style" not in st.session_state:
        st.session_state.selected_style = ""
    if "hebrew_best_sentences_each_par" not in st.session_state:
        st.session_state.hebrew_best_sentences_each_par = []  # Initialize the list
    if "restart" not in st.session_state:
        st.session_state.restart = False
    if st.session_state.done:
        placeholder = st.empty()
        create_new_story = placeholder.button('ליצירת סיפור חדש')
        if create_new_story:
            toggle_page(content="⏳", show=True)  # Add the hourglass emoji content here
            placeholder.empty()
            st.session_state.done = False  # Reset the button click state
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
    else:
        # Custom HTML with specific pixel positioning for the title and subheader
        title_html = """
            <h1 class='title_html'>
                איור לסיפור
            </h1>
        """
        subheader_html = """
            <h4 class='subheader_html'>
                צרו איורים מותאמים אישית לסיפורים בעברית
            </h4>
        """
        # Display the custom HTML titles in Streamlit
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(subheader_html, unsafe_allow_html=True)
        # Get title of the story
        title_of_story = st.text_input("שם הסיפור")
        # Get author name
        author_name = st.text_input("שם הסופר/ת")
        # Get user input in Hebrew
        user_input_hebrew = st.text_area("כתבו כאן את הסיפור שלכם לקבלת איורים מותאמים", height=225)
        # Define the options in Hebrew
        hebrew_options = [
            "ציור שמן",
            "ציור בצבעי מים",
            "שחור ולבן",
            "אימפרסיוניזם",
            "מצויר",
            "רֵיאָלִיזם",
            "פַּסטֵל",
        ]
        # Define a mapping dictionary between Hebrew and English options
        options_mapping = {
            "ציור שמן": "oil painting",
            "ציור בצבעי מים": "watercolor painting",
            "שחור ולבן": "black and white",
            "אימפרסיוניזם": "impressionism",
            "מצויר": "",
            "רֵיאָלִיזם": "realism",
            "פַּסטֵל": "pastel",
        }
        # Create the selectbox with Hebrew options for the user
        selected_style_hebrew = st.selectbox("בחרו את סגנון האיור המועדף עליכם", hebrew_options)
        # Use the mapping dictionary to get the corresponding English option in the backend
        selected_style_english = options_mapping[selected_style_hebrew]
        selected_style = selected_style_english
        translate_button = st.button("לחצו כאן ליצירת האיורים")
        # Translate only if the button is pressed or if translation is in session state
        if translate_button or st.session_state.translate:
            st.session_state.translate = True
            try:
                # Track progress
                progress_bar = st.progress(0, text="איזה כיף! עוד כמה רגעים נתחיל בהצגת האיורים")
                # Translate the input to English
                user_input_english = translate_text(user_input_hebrew)
                # st.write(user_input_english)
                # st.success(f"Translated text: {user_input_english}")
                progress_bar.progress(5, text="איזה כיף! עוד כמה רגעים נתחיל בהצגת האיורים")  # Update progress
                # Extract main characters from the translated text
                main_characters = extract_main_characters(user_input_english)
                # st.write(main_characters)
                progress_bar.progress(15, text="איזה כיף! עוד כמה רגעים נתחיל בהצגת האיורים")  # Update progress
                # Display the main characters extracted using NLTK
                # st.subheader("Main Characters:")
                # st.write(main_characters)
                resolved_text = requests.post(API_URL_CORE, json={"text": user_input_english, "main_characters": main_characters})
                if resolved_text.status_code == 200:
                    resolved_text = resolved_text.json().get("resolved_text")
                else:
                    resolved_text = user_input_english
                # st.write("resolved_texts:", resolved_text)

                # Segment the text into paragraphs
                paragraphs = segment_text(resolved_text)
                # st.write("paragraphs:", paragraphs)
                # Summarize each paragraph
                summarized_paragraphs = paragraph_summary(paragraphs)
                # st.write("summarized_paragraphs:", summarized_paragraphs)

                for par_numb in range(len(summarized_paragraphs)):
                    resolved_summarized_paragraph = requests.post(API_URL_CORE, json={"text": summarized_paragraphs[par_numb], "main_characters": main_characters})
                    if resolved_summarized_paragraph.status_code == 200:
                        resolved_summarized_paragraph = resolved_summarized_paragraph.json().get("resolved_text")
                    else:
                        resolved_summarized_paragraph = summarized_paragraphs[par_numb]
                    summarized_paragraphs[par_numb] = resolved_summarized_paragraph
                # st.write("summarized_paragraphs:", summarized_paragraphs)

                progress_bar.progress(35, text="איזה כיף! עוד כמה רגעים נתחיל בהצגת האיורים")  # Update progress
                # Generate and enhance the best sentences
                best_sentences = generate_best_sentence(summarized_paragraphs)
                # st.write("best_sentences:", best_sentences)
                hebrew_best_sentences_each_par = align_hebrew_best_sentences(best_sentences, user_input_hebrew)
                # st.write("hebrew_best_sentences_each_par:", hebrew_best_sentences_each_par)
                progress_bar.progress(55, text="איזה כיף! עוד כמה רגעים נתחיל בהצגת האיורים")  # Update progress
                # Check if best_sentences is not empty before proceeding
                if best_sentences:
                    try:
                        enhanced_text, sentence_mapping = generate_enhanced_text(best_sentences, progress_bar)
                        if enhanced_text is not None:
                            # Display the enhanced text
                            # st.subheader("Enhanced Text:")
                            # st.write(enhanced_text)
                            # st.write("")
                            # Generate general descriptions for the main characters
                            character_descriptions = generate_general_descriptions(main_characters,
                                                                                   summarized_paragraphs)
                            progress_bar.progress(100, text="תכף מסיימים!")
                            # print(f'character_descriptions: {character_descriptions}')
                            try:
                                # st.write("General Descriptions:", character_dict)
                                # Done button is now inside the main function
                                with st.form("image_selection_form"):
                                    # Modify the following lines to use classes from the CSS file
                                    st.markdown(
                                        "<h3 class='subtitle'>איורים לבחירתך שיופיעו בסיפור</h3>",
                                        unsafe_allow_html=True
                                    )
                                    st.markdown(
                                        """
                                        <style>
                                        .instructions {
                                            font-size: 18px;
                                            direction: rtl;
                                        }
                                        .svg-emoji {
                                            width: 1em;
                                            height: 1em;
                                            vertical-align: -0.1em;
                                        }
                                        </style>
                                        <h5 class='instructions'>
                                            להסרת איור לא רצוי הסירו את הסימן
                                            <svg class="svg-emoji" viewBox="0 0 24 24">
                                                <rect width="24" height="24" fill="#6a00ff"/>
                                                <path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z" fill="white"/>
                                            </svg>
                                        </h5>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                    # Call the show_enhanced_images function here
                                    model_names = [
                                        # "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
                                        "stabilityai/stable-diffusion-xl-base-1.0",
                                        "runwayml/stable-diffusion-v1-5",
                                        "black-forest-labs/FLUX.1-dev"
                                    ]
                                    with st.spinner("עוד כמה רגעים והאיור יופיע"):
                                        model_name = check_model_availability(model_names)
                                        # print(f'model_name: {model_name}')
                                        show_enhanced_images(model_name, best_sentences, sentence_mapping,
                                                             character_descriptions,
                                                             selected_style, hebrew_best_sentences_each_par)
                                    # Allow users to select images before clicking "Done"
                                    st.form_submit_button("לאחר סיום הבחירה לחצו כאן להמשך",
                                                          on_click=on_done_button_click,
                                                          args=(user_input_hebrew, hebrew_best_sentences_each_par,
                                                                title_of_story, author_name))
                            except Exception as e:
                                error_message = (
                                    f"An error occurred during text enhancement: {str(e)}"
                                )
                                # st.write(error_message)
                                st.subheader("אוי נתקלנו בבעיה בטעינת המודל המרוחק")
                                st.error("ממליצים לנסות שוב בעוד מספר רגעים")
                    except Exception as e:
                        error_message = (
                            f"An error occurred during text enhancement: {str(e)}"
                        )
                        # st.write(error_message)
                        st.subheader("אוי נתקלנו בבעיה בטעינת המודל המרוחק")
                        st.error("ממליצים לנסות שוב בעוד מספר רגעים")
                else:
                    st.error("Best sentences list is empty.")
            except Exception as e:
                error_message = (
                    f"An error occurred {str(e)}"
                )
                if len(user_input_hebrew) >= 3280:
                    st.subheader("נתקלנו בבעיה")
                    st.error("אוי! הטקסט חורג מהמגבלה של 3280 תווים, ממליצים להריץ שוב בחלקים נפרדים לפי המגבלה")
                else:
                    # st.write(error_message)
                    st.subheader("נתקלנו בבעיה")
                    st.error("אופס, נראה ששכחתם את הסיפור! אנא כתבו אותו בתיבה למעלה ונסו שוב.")


if __name__ == "__main__":
    main()
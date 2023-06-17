import streamlit as st
from PIL import Image
import math
import plotly.express as px
import base64
import io
from io import BytesIO
import os

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from docx import Document
from docx.shared import Inches
from fpdf import FPDF

# USEFUL AI LIBRARIES
import openai
import reportlab as rl
from PyPDF2 import PdfReader
import replicate
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# PROMPT ENGINEERING / CHATGPT
openai.api_key = ["Copy-paste your api here"]

openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hello ChatGPT, does this work?"}
  ]
  )

def get_completion(prompt, model='gpt-3.5-turbo'):
      messages = [{'role':'user','content':prompt}]
      response = openai.ChatCompletion.create(
          model=model,
          messages=messages,
          temperature=0)
      return response.choices[0].message['content']

def prompt_text(sentence):
  prompt = f"""
    Reestructura el texto de forma esquemática con las siguientes condiciones:
    - Reestructura los párrafos y hazlos más pequeños.
    - Adáptalo al público general.
    - Añade más signos de puntuación.
    - Elabora una frase que concluya el texto.
    ```{sentence}```"""
  return get_completion(prompt)

def prompt_image(sentence):
  prompt = f"""
  Redacta una frase corta en INGLÉS en la que indiques qué imagen podría
  acompañar a este texto y su descripción para una inteligencia artificial,
  omite las palabras previas a tu respuesta y devuélve solo el contenido
  de la imagen. Acaba la frase con las palabras realista, fondo plano: \n
  {sentence}"""

  response = get_completion(prompt)

  scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
  pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
  pipe = pipe.to("cuda")
  image = pipe(response).images[0]

  return image

# STABLE DIFFUSSION
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# CREATE AND VISUALIZE DOCS

def create_pdf(filename, text, image):
  max_width = 450
  c = canvas.Canvas(filename)
  colores = {'b': colors.red, 'd': colors.blue, 'p': colors.green, 'q': colors.magenta,
              'm': colors.plum, 'n': colors.orange}

  # Set the font and font size
  pdfmetrics.registerFont(TTFont('verdana', 'https://github.com/matomo-org/travis-scripts/raw/master/fonts/Verdana.ttf'))
  c.setFont("verdana", 12)

  # Set the position where the text should start (x, y)
  x = 80
  y = 750

  # Wrap the text into lines
  lines = rl.lib.utils.simpleSplit(text, c._fontname, c._fontsize, max_width)

  # Write each line onto the PDF
  for line in lines:

      for i, letra in enumerate(line):
          color = colores.get(letra.lower(), colors.black)
          c.setFillColor(color)
          c.drawString(x, y, letra)
          x += c.stringWidth(letra, "verdana", 12)  # Utilizar Verdana como tipo de letra

      if y < 50:
        c.showPage()
        y = 750 + 14

      x = 80
      y -= 14 # Adjust line spacing as needed


  c.drawInlineImage(image, x=200, y=y-200, width=200, height=200)

  # Save the canvas to the PDF file
  c.save()

def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# APP
_, cent_co, _ = st.columns(3)
with cent_co:
    st.image('https://raw.githubusercontent.com/solor5/dislexIA/main/logo.png')

greetings = '<p style="font-family:Verdana; color:Black; font-size: 18px;">¡Bienvenid@ a DislexIA!</p>'
st.markdown(greetings, unsafe_allow_html=True)

description = '<p style="font-family:Verdana; color:Black; font-size: 16px;">Esta herramienta permite reestructurar y adaptar cualquier tipo de texto para que su lectura resulte más cómoda para los usuarios con dislexia. Su funcionamiento se estructura sobre modelos de procesamiento de lenguaje natural y modelos de generación de imágenes. Toda la información que se introduzca en esta herramienta será procesada por ChatGPT y StableDiffusion 1.5.</p>'
st.markdown(description, unsafe_allow_html=True)

authors = '<p style="font-family:Verdana; color:Black; font-size: 16px;">Proyecto elaborado por <a href="https://www.linkedin.com/in/marinatouceda/">Mariña Touceda</a>, <a href="https://www.linkedin.com/in/diana-araujo-morera-767ba283/">Diana Araujo</a>, <a href="add">Marina García</a> y <a href="https://www.linkedin.com/in/william-solórzano/">William Solórzano</a></p>'
st.markdown(authors, unsafe_allow_html=True)

input_mode = st.selectbox(
        "Seleccione el formato de entrada:",
        ("Entrada de texto", "PDF")
)

if input_mode == "Entrada de texto":
  text = st.text_area('Coloque su texto aquí')
  if st.button('Ejecutar'):
    wait = '<p style="font-family:Verdana; color:Black; font-size: 14px; text-align: center;"> Paciencia, Roma no se construyó en un día y ChatGPT tampoco. Nuestro modelo no será conocido por ser el más rápido pero si el más MOLÓN... intenta pasarte un nivel del CandyCrush mientras esperas por tu texto :P </p>'
    st.markdown(wait, unsafe_allow_html=True)
    response = prompt_text(text)
    str = '<p style="font-family:Verdana; color:Black; font-size: 12px; text-align: center;">--- PDF procesado ---</p>'
    st.markdown(str, unsafe_allow_html=True)
    image = prompt_image(response)
    str = '<p style="font-family:Verdana; color:Black; font-size: 12px; text-align: center;">--- Imagen creada ---</p>'
    st.markdown(str, unsafe_allow_html=True)

    filename =  "output.pdf"
    create_pdf(filename, response, image)
    show_pdf(filename)

elif input_mode == "PDF":
  uploaded_pdf = st.file_uploader('Importe el archivo PDF', type="pdf")

  if st.button('Ejecutar'):
    pdf_reader = PdfReader(uploaded_pdf)
    text = ''
    for i in range(len(pdf_reader.pages)):
      page_file = pdf_reader.pages[i].extract_text()
      text += page_file

    wait = '<p style="font-family:Verdana; color:Black; font-size: 14px; text-align: center;"> Paciencia, Roma no se construyó en un día y ChatGPT tampoco. Nuestro modelo no será conocido por ser el más rápido pero si el más MOLÓN... intenta pasarte un nivel del CandyCrush mientras esperas por tu texto :P </p>'
    st.markdown(wait, unsafe_allow_html=True)
    str = '<p style="font-family:Verdana; color:Black; font-size: 12px; text-align: center;">--- PDF leído ---</p>'
    st.markdown(str, unsafe_allow_html=True)


    response = prompt_text(text)
    str = '<p style="font-family:Verdana; color:Black; font-size: 12px; text-align: center;">--- PDF procesado ---</p>'
    st.markdown(str, unsafe_allow_html=True)
    image = prompt_image(response)
    str = '<p style="font-family:Verdana; color:Black; font-size: 12px; text-align: center;">--- Imagen creada ---</p>'
    st.markdown(str, unsafe_allow_html=True)

    filename =  "output.pdf"
    create_pdf(filename, response, image)
    show_pdf(filename)

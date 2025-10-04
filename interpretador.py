import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import openai
from openai import OpenAI

# Crea el cliente con la API key de Streamlit
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])



@st.cache_resource

def load_models():
    model_style = tf.keras.models.load_model("StyleClass.keras", compile=False)
    model_category = tf.keras.models.load_model("CategoryClass.keras", compile=False)
    model_emotions = tf.keras.models.load_model("EmotionClass.keras", compile=False)
    return model_style, model_category, model_emotions

model_style, model_category, model_emotions = load_models()

def preprocess_image(image, target_size=(128, 128)):
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_palette(image, n_colors=5):
    img = image.resize((150, 150))  # redimensiona para acelerar
    img_array = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(img_array)
    colors = kmeans.cluster_centers_.astype(int)

    # Mostrar paleta
    fig, ax = plt.subplots(figsize=(8, 2))
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=np.array(color)/255))
    ax.set_xlim(0, n_colors)
    ax.set_ylim(0, 1)
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return colors, buf

def generate_interpretation(emotions, style, category, keywords, description=""):
    prompt = f"""
    Analiza esta pintura. 
    - Estilo: {style}
    - Categoría: {category}
    - Emociones detectadas: {emotions}
    - Palabras clave: {', '.join(keywords)}

    {description}

    Por favor escribe una breve interpretación poética de la obra, conectando emociones y estilo.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    
    return response.choices[0].message.content

# --------------------
# INTERFAZ STREAMLIT
# --------------------
st.title("🎨 Interpretador de Pinturas con IA")
st.write("Sube una pintura y descubre su estilo, categoría, emociones, paleta de colores y una interpretación generada por IA.")

uploaded_file = st.file_uploader("📂 Sube tu pintura", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_container_width=True)

    # Preprocesar
    img_array = preprocess_image(image)

    # 🔹 Predicciones
    pred_style = model_style.predict(img_array)
    pred_category = model_category.predict(img_array)
    pred_emotions = model_emotions.predict(img_array)

    # Etiquetas (ejemplo, debes cambiarlas por tus etiquetas reales)
    styles = ['Modern Art', 'Post Renaissance Art', 'Contemporary Art','Renaissance Art']
    categories = ['Impressionism', 'Neo-Expressionism', 'Post-Impressionism',
       'Cubism', 'Romanticism', 'Expressionism', 'Realism', 'Minimalism',
       'Pop Art', 'Rococo', 'Color Field Painting', 'Early Renaissance',
       'Neoclassicism', 'Art Informel', 'Baroque', 'Abstract Art',
       'Lyrical Abstraction', 'Surrealism', 'Abstract Expressionism',
       'Magic Realism', 'Northern Renaissance', 'High Renaissance']
    emotions = [ 'agreeableness', 'anger','anticipation', 'arrogance','disagreeableness', 
        'disgust', 'fear','gratitude', 'happiness', 'humility','love', 'optimism', 
        'pessimism','regret', 'sadness', 'shame','shyness', 'surprise']

    style_result = styles[np.argmax(pred_style)]
    category_result = categories[np.argmax(pred_category)]
    emotion_result = emotions[np.argmax(pred_emotions)]

    # 🔹 Paleta de colores
    colors, buf = get_palette(image)
    st.subheader("Paleta de colores")
    st.image(buf, use_container_width=False)

    # 🔹 Keywords (puedes refinar esta lista)
    keywords = [style_result, category_result, emotion_result]

    # 🔹 Interpretación con OpenAI
    st.subheader("Interpretación generada por IA")
    interpretation ="Interpretación de prueba: la obra transmite emociones intensas y un estilo único."
    st.write(interpretation)

    # 🔹 Resultados en texto
    st.subheader("Resultados del análisis")
    st.write(f"**Estilo detectado:** {style_result}")
    st.write(f"**Categoría detectada:** {category_result}")
    st.write(f"**Emoción detectada:** {emotion_result}")
    st.write(f"**Palabras clave:** {', '.join(keywords)}")

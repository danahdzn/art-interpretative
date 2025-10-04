import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from openai import OpenAI

# -------------------
# CONFIGURACIÓN DE LA PÁGINA Y ESTILO
# -------------------
st.set_page_config(
    page_title="🎨 Art Interpretative",
    page_icon="🖌️",
    layout="wide"
)

# CSS para fondo rosita y estilo
st.markdown(
    """
    <style>
    /* Fondo degradado de varios tonos de rosa */
    .stApp {
        background: linear-gradient(135deg, #ffe6f0, #ffd6eb, #ffcce5, #ffb3da);
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    img = image.resize((150, 150))
    img_array = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(img_array)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

def display_palette(colors):
    cols = st.columns(len(colors))
    for i, color in enumerate(colors):
        cols[i].markdown(
            f"<div style='background-color: rgb{tuple(color)}; height: 50px; border-radius: 5px;'></div>",
            unsafe_allow_html=True
        )

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
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content
    except:
        return "Interpretación de prueba: la obra transmite emociones intensas y un estilo único."

# -------------------
# INTERFAZ PRINCIPAL
# -------------------
st.markdown(
    """
    <style>
    /* Fondo degradado rosa pastel */
    .stApp {
        background: linear-gradient(135deg, #ffe6f0, #ffd6eb, #ffcce5, #ffb3da);
    }

    /* Encabezado */
    .custom-header {
        background-color: #FB9EC6; /* rosa intenso */
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    .custom-header h1 {
        color: white;
        font-size: 2.5em;
        font-family: 'Trebuchet MS', sans-serif;
        margin: 0;
    }
    </style>

    <div class="custom-header">
        <h1>🎨 Art Interpretative 🌸</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("💡 Upload your picture")

uploaded_file = st.file_uploader("📂 Sube tu pintura", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col_img, col_results = st.columns([2, 1])

    with col_img:
        st.image(image, caption="Imagen subida", use_container_width=False, width=400)

    with col_results:
        img_array = preprocess_image(image)
        pred_style = model_style.predict(img_array)
        pred_category = model_category.predict(img_array)
        pred_emotions = model_emotions.predict(img_array)

        # Etiquetas (ajusta según tus modelos)
        styles = ['Contemporary Art', 'Modern Art', 'Post Renaissance Art','Renaissance Art']
        categories = ['Abstract Art', 'Abstract Expressionism', 'Art Informel', 'Baroque', 
        'Color Field Painting', 'Cubism', 'Early Renaissance', 'Expressionism', 'High Renaissance',
        'Impressionism', 'Lyrical Abstraction', 'Magic Realism', 'Minimalism', 'Neo-Expressionism', 
        'Neoclassicism', 'Northern Renaissance', 'Pop Art', 'Post-Impressionism', 'Realism', 'Rococo', 'Romanticism', 'Surrealism']
        emotions_list = ['agreeableness', 'anger','anticipation', 'arrogance','disagreeableness', 
            'disgust', 'fear','gratitude', 'happiness', 'humility','love', 'optimism', 
            'pessimism','regret', 'sadness', 'shame','shyness', 'surprise']

        style_result = styles[np.argmax(pred_style)]
        category_result = categories[np.argmax(pred_category)]
        emotion_result = emotions_list[np.argmax(pred_emotions)]

        # Paleta de colores
        colors = get_palette(image)
        st.subheader("🎨 Paleta de colores")
        display_palette(colors)

        # Keywords
        keywords = [style_result, category_result, emotion_result]

        # Interpretación en expander
        with st.expander("🖌 Interpretación generada por IA"):
            interpretation = generate_interpretation(emotion_result, style_result, category_result, keywords)
            st.write(interpretation)

        # Resultados
        st.subheader("📊 Resultados del análisis")
        st.markdown(f"**Estilo detectado:** {style_result}")
        st.markdown(f"**Categoría detectada:** {category_result}")
        st.markdown(f"**Emoción detectada:** {emotion_result}")
        st.markdown(f"**Palabras clave:** {', '.join(keywords)}")

        # Botón de descarga
        st.download_button(
            "📥 Descargar resultados",
            data=f"Estilo: {style_result}\nCategoría: {category_result}\nEmoción: {emotion_result}\nKeywords: {', '.join(keywords)}\nInterpretación: {interpretation}",
            file_name="resultados_pintura.txt"
        )

import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile
import os

# --- 1. Configuración Visual (SIEMPRE debe ir primero) ---
st.set_page_config(page_title="Biometría Local | Compliance", page_icon="👤")

# --- 2. SISTEMA DE SEGURIDAD (LOGIN) ---
def check_password():
    """Devuelve True si el usuario ingresó la contraseña correcta."""
    
    def password_entered():
        # Comprueba si el usuario existe en nuestros "Secrets" y si la contraseña coincide
        user = st.session_state["username"]
        pwd = st.session_state["password"]
        
        if user in st.secrets["passwords"] and st.secrets["passwords"][user] == pwd:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Por seguridad, borramos la contraseña de la memoria
        else:
            st.session_state["password_correct"] = False

    # Si es la primera vez que entra a la página
    if "password_correct" not in st.session_state:
        st.title("🔒 Acceso Restringido")
        st.info("Herramienta de uso interno. Por favor, identifíquese.")
        st.text_input("Usuario", key="username")
        st.text_input("Contraseña", type="password", key="password")
        st.button("Ingresar", on_click=password_entered, type="primary")
        return False
        
    # Si intentó entrar y falló
    elif not st.session_state["password_correct"]:
        st.title("🔒 Acceso Restringido")
        st.text_input("Usuario", key="username")
        st.text_input("Contraseña", type="password", key="password")
        st.button("Ingresar", on_click=password_entered, type="primary")
        st.error("❌ Usuario o contraseña incorrectos. Intente nuevamente.")
        return False
        
    # Si la contraseña es correcta
    else:
        return True

# --- 3. APLICACIÓN PRINCIPAL ---
# Todo este bloque solo se ejecuta SI la función check_password() es Verdadera
if check_password():
    
    # Botón para cerrar sesión en la barra lateral
    st.sidebar.title("Opciones")
    if st.sidebar.button("🚪 Cerrar Sesión"):
        del st.session_state["password_correct"]
        st.rerun()

    st.markdown("## 👤 Verificador Biométrico de Identidad")
    st.info("Comparación de rostros mediante IA (Modelo VGG-Face).")

    # --- Carga de Imágenes ---
    col1, col2 = st.columns(2)
    with col1:
        f1 = st.file_uploader("Foto A (ID/Cédula)", type=['jpg', 'jpeg', 'png'])
        if f1: st.image(Image.open(f1), caption="Referencia")

    with col2:
        f2 = st.file_uploader("Foto B (Selfie)", type=['jpg', 'jpeg', 'png'])
        if f2: st.image(Image.open(f2), caption="Sujeto")

    # --- Lógica de Análisis ---
    if st.button('🚀 Ejecutar Verificación', type="primary"):
        if f1 and f2:
            with st.spinner('Procesando rasgos faciales...'):
                t1 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                t2 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                
                try:
                    Image.open(f1).convert('RGB').save(t1.name)
                    Image.open(f2).convert('RGB').save(t2.name)

                    # Comparación Biométrica
                    res = DeepFace.verify(
                        img1_path=t1.name, 
                        img2_path=t2.name, 
                        enforce_detection=False,
                        model_name="VGG-Face"
                    )
                    
                    distancia = res["distance"]
                    similitud = (1 - distancia) * 100
                    
                    st.divider()
                    if res["verified"]:
                        st.success(f"### ✅ IDENTIDAD VERIFICADA")
                        st.metric("Grado de Similitud", f"{similitud:.2f}%")
                    else:
                        st.error(f"### ❌ IDENTIDAD NO COINCIDE")
                        st.metric("Grado de Similitud", f"{similitud:.2f}%")
                    
                    with st.expander("Ver rastro técnico del análisis"):
                        st.json(res)

                except Exception as e:
                    st.error(f"Error técnico: {e}")
                finally:
                    t1.close(); t2.close()
                    if os.path.exists(t1.name): os.remove(t1.name)
                    if os.path.exists(t2.name): os.remove(t2.name)
        else:
            st.warning("Debe cargar ambas imágenes para realizar el análisis.")

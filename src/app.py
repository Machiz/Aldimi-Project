import os
import sys
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from sqlalchemy import func

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.database.connection import SessionLocal
from src.database.models import PacienteEvaluacionOCR, PacientePrediccionPrioridad, InsumoInventario, InsumoPrediccion
from src.pipeline.data_pipeline import AldimiPipeline

# CONFIGURACIÓN DE PÁGINA STREAMLIT
st.set_page_config(
    page_title="ALDIMI-Predict | Dashboard de Decisiones",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ESTILOS CSS PERSONALIZADOS (Aesthetics: Premium, Modern, Dark Mode inspired accent elements)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B, #8B0000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border-left: 5px solid #FF4B4B;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2b2b2b;
    }
    
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888;
    }
    
    .badge {
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        text-align: center;
    }
    
    .badge-alto {
        background-color: #F8D7DA;
        color: #721C24;
        border: 1px solid #F5C6CB;
    }
    
    .badge-medio {
        background-color: #FFF3CD;
        color: #856404;
        border: 1px solid #FFEBAA;
    }
    
    .badge-bajo {
        background-color: #D4EDDA;
        color: #155724;
        border: 1px solid #C3E6CB;
    }
    
    /* Traffic Lights Styles */
    .traffic-light-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    
    .traffic-dot {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        display: inline-block;
    }
    
    .dot-red {
        background-color: #ff3b30;
        box-shadow: 0 0 10px #ff3b30;
    }
    
    .dot-orange {
        background-color: #ff9500;
        box-shadow: 0 0 10px #ff9500;
    }
    
    .dot-green {
        background-color: #34c759;
        box-shadow: 0 0 10px #34c759;
    }
</style>
""", unsafe_allow_html=True)

# INICIALIZAR PIPELINE
@st.cache_resource
def get_pipeline():
    return AldimiPipeline()

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Error al inicializar el pipeline predictivo: {e}")
    st.info("Asegúrese de correr `python3 src/database/initialize_data.py` primero para entrenar los modelos y estructurar la base de datos.")
    st.stop()

# LOG HISTÓRICO DE CONSUMO
@st.cache_data
def load_historical_consumption():
    csv_path = Path("data/insumos_cancer_final.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df
    return pd.DataFrame()

df_history = load_historical_consumption()

# SESION DB
db = SessionLocal()

# OBTENER PREDICCIONES DE INSUMOS MÁS RECIENTES
def get_latest_insumo_predictions(db):
    max_date = db.query(func.max(InsumoPrediccion.fecha_prediccion)).scalar()
    if not max_date:
        # Si no hay predicciones calculadas, ejecutamos el forecast inicial
        pipeline.run_demand_forecast(db)
        max_date = db.query(func.max(InsumoPrediccion.fecha_prediccion)).scalar()
        
    if max_date:
        # Traer todo el lote de predicciones de la última corrida (dentro de un margen de 10s)
        threshold_date = max_date - datetime.timedelta(seconds=10)
        preds = db.query(InsumoPrediccion).filter(InsumoPrediccion.fecha_prediccion >= threshold_date).all()
    else:
        preds = []
    return {p.item_id: p for p in preds}

# BARRA LATERAL - NAVEGACIÓN
st.sidebar.markdown("<h2 style='text-align: center;'>ALDIMI-Predict</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
view_mode = st.sidebar.radio(
    "Seleccione Vista:",
    [
        "🩺 Gestión de Pacientes",
        "📦 Logística e Inventario",
        "⚠️ Módulo de Alertas Críticas"
    ]
)

st.sidebar.markdown("<br><br><br><hr>", unsafe_allow_html=True)
st.sidebar.caption("Ecosistema Predictivo ALDIMI - Versión 1.0 (CRISP-DM Fase 6)")
st.sidebar.caption("Fecha simulación: 10 de Febrero, 2026")

# OBTENER DATOS ACTUALES
latest_preds = get_latest_insumo_predictions(db)
active_patients_count = db.query(PacienteEvaluacionOCR).count()

# ==============================================================================
# VISTA 1: GESTIÓN DE PACIENTES
# ==============================================================================
if view_mode == "🩺 Gestión de Pacientes":
    st.markdown("<div class='main-title'>Gestión de Pacientes</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Ingreso y clasificación en tiempo real de pacientes oncológicos</div>", unsafe_allow_html=True)
    
    col_form, col_list = st.columns([1, 1])
    
    with col_form:
        st.markdown("### 📥 Registrar Nuevo Paciente")
        
        with st.form("new_patient_form", clear_on_submit=True):
            # Información Básica
            c1, c2 = st.columns(2)
            eval_id = c1.text_input("ID Evaluación (OCR)", value=f"OCR-{np.random.randint(1000, 9999)}")
            paciente_id = c2.text_input("ID Paciente", value=f"P-{np.random.randint(10000, 99999)}")
            
            c3, c4, c5 = st.columns(3)
            edad = c3.number_input("Edad", min_value=0, max_value=120, value=55)
            sexo = c4.selectbox("Sexo", ["femenino", "masculino"])
            region = c5.text_input("Región", value="Lima")
            
            c6, c7 = st.columns(2)
            zona = c6.selectbox("Zona Procedencia", ["urbana", "rural"])
            viaja_prov = c7.selectbox("Viaja desde Provincia", ["si", "no"])
            
            # Variables de vulnerabilidad y clínicas
            c8, c9 = st.columns(2)
            nivel_vulnerabilidad = c8.selectbox("Nivel Vulnerabilidad", ["bajo", "medio", "alto", "muy alto"])
            dificultad_acceso = c9.selectbox("Dificultad Acceso Salud", ["baja", "media", "alta"])
            
            acompanante = st.selectbox("Acompañante", ["si", "no"])
            
            st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
            st.markdown("**Estado Clínico del Paciente**")
            
            c10, c11 = st.columns(2)
            tipo_histologico = c10.text_input("Tipo Histológico", value="adenocarcinoma")
            estadio = c11.selectbox("Estadio Cáncer", ["Estadio 0", "Estadio I", "Estadio II", "Estadio III", "Estadio IV"])
            
            c12, c13 = st.columns(2)
            grado_histologico = c12.selectbox("Grado Histológico", ["bien diferenciado", "moderadamente diferenciado", "poco diferenciado", "desconocido"])
            profundidad_tumor = c13.selectbox("Profundidad Tumor (TNM)", ["T1", "T2", "T3", "T4"])
            
            c14, c15 = st.columns(2)
            ganglios = c14.number_input("Ganglios Afectados", min_value=0, max_value=100, value=0)
            metastasis = c15.selectbox("Metástasis", ["no", "si"])
            sitio_meta = st.text_input("Sitio Metástasis", value="ninguno")
            
            st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
            st.markdown("**Síntomas y Tratamientos Requeridos**")
            
            c16, c17, c18 = st.columns(3)
            perdida_peso = c16.number_input("Pérdida peso (kg)", min_value=0.0, max_value=50.0, value=0.0)
            estado_nutri = c17.selectbox("Estado Nutricional", ["bueno", "riesgo desnutricion", "desnutricion"])
            hemoglobina = c18.number_input("Hemoglobina (g/dL)", min_value=3.0, max_value=25.0, value=12.0)
            
            c19, c20, c21 = st.columns(3)
            anemia = c19.selectbox("Anemia", ["no", "si"])
            dolor = c20.selectbox("Dolor", ["ausente", "leve", "moderado", "severo"])
            vomitos = c21.selectbox("Vómitos frecuentes", ["no", "si"])
            
            c22, c23, c24 = st.columns(3)
            sangrado = c22.selectbox("Sangrado digestivo", ["no", "si"])
            fatiga = c23.selectbox("Fatiga", ["no", "si"])
            dificultad_alimentar = c24.selectbox("Dificultad alimentarse", ["no", "si"])
            
            c25, c26 = st.columns(2)
            comorbilidades = c25.selectbox("Comorbilidades", ["no", "si"])
            estado_funcional = c26.selectbox("Estado Funcional", ["totalmente activo", "limitado", "ambulatorio", "dependiente en cama", "desconocido"])
            
            c27, c28 = st.columns(2)
            req_cirugia = c27.selectbox("Requiere Cirugía", ["no", "si"])
            req_quimio = c28.selectbox("Requiere Quimioterapia", ["no", "si"])
            
            c29, c30 = st.columns(2)
            req_soporte = c29.selectbox("Requiere Soporte Nutricional", ["no", "si"])
            req_paliativos = c30.selectbox("Requiere Cuidados Paliativos", ["no", "si"])
            
            tratamiento_principal = st.text_input("Tratamiento Principal", value="quimioterapia")
            
            submit_btn = st.form_submit_button("🩺 Evaluar y Registrar Paciente")
            
            if submit_btn:
                # Validar fechas (simulamos ingreso y diagnóstico razonables)
                today_date = datetime.date(2026, 2, 10)
                record_data = {
                    "evaluacion_id": eval_id,
                    "paciente_id": paciente_id,
                    "fecha_evaluacion": today_date.strftime("%Y-%m-%d"),
                    "fecha_diagnostico": (today_date - datetime.timedelta(days=90)).strftime("%Y-%m-%d"),
                    "fecha_ingreso_aldimi": (today_date - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
                    "edad": edad,
                    "sexo": sexo,
                    "region_procedencia": region,
                    "zona_procedencia": zona,
                    "viaja_desde_provincia": viaja_prov,
                    "acompanante": acompanante,
                    "nivel_vulnerabilidad": nivel_vulnerabilidad,
                    "dificultad_acceso_salud": dificultad_acceso,
                    "tipo_histologico": tipo_histologico,
                    "estadio": estadio,
                    "grado_histologico": grado_histologico,
                    "profundidad_tumor": profundidad_tumor,
                    "ganglios_afectados": ganglios,
                    "metastasis": metastasis,
                    "sitio_metastasis": sitio_meta,
                    "perdida_peso_kg": perdida_peso,
                    "estado_nutricional": estado_nutri,
                    "hemoglobina": hemoglobina,
                    "anemia": anemia,
                    "dolor": dolor,
                    "vomitos_frecuentes": vomitos,
                    "sangrado_digestivo": sangrado,
                    "fatiga": fatiga,
                    "dificultad_alimentarse": dificultad_alimentar,
                    "comorbilidades": comorbilidades,
                    "estado_funcional": estado_funcional,
                    "requiere_cirugia": req_cirugia,
                    "requiere_quimioterapia": req_quimio,
                    "requiere_soporte_nutricional": req_soporte,
                    "requiere_cuidados_paliativos": req_paliativos,
                    "tratamiento_principal": tratamiento_principal
                }
                
                try:
                    # Ejecutar pipeline secuencial
                    pred_res, projections = pipeline.run_pipeline_for_new_patient(record_data, date_str="2026-02-10")
                    st.success(f"¡Paciente {paciente_id} ingresado y clasificado con éxito!")
                    
                    # Mostrar alerta emergente del resultado
                    prior = pred_res.prioridad_predicha.upper()
                    prob_alto = pred_res.probabilidad_alto * 100
                    
                    if prior == "ALTO":
                        st.error(f"🚨 CLASIFICACIÓN DE ALTA PRIORIDAD (Probabilidad: {prob_alto:.1f}%)")
                    elif prior == "MEDIO":
                        st.warning(f"⚠️ CLASIFICACIÓN DE PRIORIDAD MEDIA (Probabilidad: {pred_res.probabilidad_medio*100:.1f}%)")
                    else:
                        st.success(f"✅ CLASIFICACIÓN DE PRIORIDAD BAJA (Probabilidad: {pred_res.probabilidad_bajo*100:.1f}%)")
                        
                    # Recargar la página para refrescar tablas
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error al ejecutar el pipeline: {ex}")

    with col_list:
        st.markdown(f"### 📋 Censo de Pacientes Activos ({active_patients_count} pacientes)")
        
        # Cargar tabla de pacientes
        patients_list = db.query(
            PacienteEvaluacionOCR.paciente_id,
            PacienteEvaluacionOCR.edad,
            PacienteEvaluacionOCR.sexo,
            PacienteEvaluacionOCR.estadio,
            PacientePrediccionPrioridad.prioridad_predicha
        ).join(
            PacientePrediccionPrioridad, PacienteEvaluacionOCR.evaluacion_id == PacientePrediccionPrioridad.evaluacion_id
        ).order_by(PacientePrediccionPrioridad.fecha_prediccion.desc()).all()
        
        if patients_list:
            df_p = pd.DataFrame(patients_list, columns=["ID Paciente", "Edad", "Sexo", "Estadio", "Prioridad"])
            
            # Aplicar badges coloreadas usando un formateador HTML o dataframes interactivos
            def color_prioridad(val):
                if val == 'alto':
                    return 'background-color: #ffcccc; color: #cc0000; font-weight: bold;'
                elif val == 'medio':
                    return 'background-color: #ffe6b3; color: #cc7a00;'
                else:
                    return 'background-color: #d1f2d9; color: #1e7a34;'
                    
            st.dataframe(
                df_p.style.map(color_prioridad, subset=['Prioridad']),
                use_container_width=True,
                height=600
            )
        else:
            st.info("No hay pacientes registrados en el censo.")

# ==============================================================================
# VISTA 2: LOGÍSTICA E INVENTARIO
# ==============================================================================
elif view_mode == "📦 Logística e Inventario":
    st.markdown("<div class='main-title'>Logística e Inventario</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Proyecciones de demanda e histórico de consumo de insumos críticos</div>", unsafe_allow_html=True)
    
    # Obtener nombres de insumos
    insumos_list = db.query(InsumoInventario.item_id, InsumoInventario.item_nombre).all()
    insumo_options = {f"{i.item_id} - {i.item_nombre}": i.item_id for i in insumos_list}
    
    selected_option = st.selectbox("Seleccione Insumo Clínico / Oncológico:", list(insumo_options.keys()))
    selected_item_id = insumo_options[selected_option]
    
    insumo_data = db.query(InsumoInventario).filter_by(item_id=selected_item_id).first()
    pred_data = latest_preds.get(selected_item_id)
    
    if insumo_data and pred_data:
        # Métricas principales
        c1, c2, c3, c4 = st.columns(4)
        
        c1.markdown(f"""
        <div class='card'>
            <div class='metric-label'>Stock Actual</div>
            <div class='metric-value'>{insumo_data.stock_actual:.0f}</div>
            <div style='color: #666; font-size: 0.8rem;'>Mínimo Requerido: {insumo_data.stock_minimo:.0f} {insumo_data.unidad_medida}</div>
        </div>
        """, unsafe_allow_html=True)
        
        c2.markdown(f"""
        <div class='card' style='border-left-color: #ff9500;'>
            <div class='metric-label'>Consumo Diario</div>
            <div class='metric-value'>{insumo_data.consumo_diario:.1f}</div>
            <div style='color: #666; font-size: 0.8rem;'>Promedio 7 días: {insumo_data.consumo_promedio_7d:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        c3.markdown(f"""
        <div class='card' style='border-left-color: #34c759;'>
            <div class='metric-label'>Demanda Proyectada 7d</div>
            <div class='metric-value'>{pred_data.demanda_proyectada_7d:.1f}</div>
            <div style='color: #666; font-size: 0.8rem;'>En {insumo_data.unidad_medida}</div>
        </div>
        """, unsafe_allow_html=True)
        
        c4.markdown(f"""
        <div class='card' style='border-left-color: #007aff;'>
            <div class='metric-label'>Demanda Proyectada 14d</div>
            <div class='metric-value'>{pred_data.demanda_proyectada_14d:.1f}</div>
            <div style='color: #666; font-size: 0.8rem;'>En {insumo_data.unidad_medida}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Gráfico Histórico y Proyectado
        st.markdown("### 📊 Historial de Consumo y Proyecciones de Demanda")
        
        if not df_history.empty:
            df_item_history = df_history[df_history['item_id'] == selected_item_id].copy()
            df_item_history = df_item_history.sort_values(by='fecha')
            
            # Crear gráfico con Plotly
            fig = go.Figure()
            
            # Histórico
            fig.add_trace(go.Scatter(
                x=df_item_history['fecha'],
                y=df_item_history['consumo_diario'],
                mode='lines',
                name='Consumo Diario Real',
                line=dict(color='#FF4B4B', width=2)
            ))
            
            # Stock histórico
            fig.add_trace(go.Scatter(
                x=df_item_history['fecha'],
                y=df_item_history['stock_actual'],
                mode='lines',
                name='Nivel de Stock Real',
                line=dict(color='#888', width=1.5, dash='dash')
            ))
            
            # Agregar predicciones futuras (simuladas al final del gráfico)
            last_date = df_item_history['fecha'].max()
            future_dates = [last_date + datetime.timedelta(days=7), last_date + datetime.timedelta(days=14)]
            future_demands = [pred_data.demanda_proyectada_7d, pred_data.demanda_proyectada_14d]
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_demands,
                mode='markers+lines',
                name='Demanda Proyectada (Model ML)',
                marker=dict(size=10, color='#007aff'),
                line=dict(color='#007aff', width=2, dash='dot')
            ))
            
            # Stock Mínimo Horizontal
            fig.add_hline(
                y=insumo_data.stock_minimo,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Stock Mínimo ({insumo_data.stock_minimo:.0f})",
                annotation_position="top left"
            )
            
            fig.update_layout(
                title=f"Historial y Proyecciones para {insumo_data.item_nombre.upper()}",
                xaxis_title="Fecha",
                yaxis_title="Cantidad",
                legend=dict(x=0.01, y=0.99),
                hovermode="x unified",
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se encontró el log histórico para graficar.")

# ==============================================================================
# VISTA 3: MÓDULO DE ALERTAS CRÍTICAS
# ==============================================================================
elif view_mode == "⚠️ Módulo de Alertas Críticas":
    st.markdown("<div class='main-title'>Módulo de Alertas de Stock Crítico</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Semáforo de cobertura de insumos oncológicos en el albergue</div>", unsafe_allow_html=True)
    
    # 1. ALERTAS EXPLÍCITAS EN CUADROS COLOREADOS
    col_7d, col_14d = st.columns(2)
    
    items_criticos_7d = []
    items_criticos_14d = []
    
    for item_id, pred in latest_preds.items():
        if pred.stock_critico_7d == 1:
            items_criticos_7d.append(pred.item_nombre)
        if pred.stock_critico_14d == 1:
            items_criticos_14d.append(pred.item_nombre)
            
    with col_7d:
        st.markdown("### 🔴 Agotamiento Crítico en los Próximos 7 Días")
        if items_criticos_7d:
            for item in items_criticos_7d:
                st.error(f"🚨 **{item.upper()}**: Stock insuficiente para cubrir la demanda proyectada a 7 días.")
        else:
            st.success("✅ Ningún insumo corre riesgo de agotarse en los próximos 7 días.")
            
    with col_14d:
        st.markdown("### 🟡 Agotamiento en los Próximos 14 Días")
        if items_criticos_14d:
            for item in items_criticos_14d:
                st.warning(f"⚠️ **{item.upper()}**: Stock insuficiente para la demanda proyectada a 14 días (Anticipar compra/donación).")
        else:
            st.success("✅ Ningún insumo corre riesgo de agotarse en los próximos 14 días.")
            
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("### 🚥 Semáforo de Cobertura General")
    
    # Tabla con semáforo
    tabla_data = []
    for item_id, pred in latest_preds.items():
        # Obtener datos de inventario
        inv = db.query(InsumoInventario).filter_by(item_id=item_id).first()
        if not inv:
            continue
            
        cobertura = pred.dias_cobertura
        if cobertura < 7.0:
            sem_html = "<span class='traffic-dot dot-red'></span> <b>Crítico (&lt;7 días)</b>"
            estado = "Rojo"
        elif 7.0 <= cobertura < 14.0:
            sem_html = "<span class='traffic-dot dot-orange'></span> <b>Riesgo (7-14 días)</b>"
            estado = "Naranja"
        else:
            sem_html = "<span class='traffic-dot dot-green'></span> <b>Seguro (&gt;14 días)</b>"
            estado = "Verde"
            
        tabla_data.append({
            "ID Insumo": inv.item_id,
            "Nombre Insumo": inv.item_nombre.upper(),
            "Categoría": inv.categoria_item,
            "Stock Actual": inv.stock_actual,
            "Consumo Diario": inv.consumo_diario,
            "Demanda 7d Proy": pred.demanda_proyectada_7d,
            "Demanda 14d Proy": pred.demanda_proyectada_14d,
            "Días Cobertura": cobertura,
            "Estado Cobertura": sem_html,
            "Status": estado
        })
        
    df_alertas = pd.DataFrame(tabla_data)
    
    # Filtro
    filtro_estado = st.multiselect("Filtrar por Semáforo:", ["Rojo", "Naranja", "Verde"], default=["Rojo", "Naranja", "Verde"])
    df_filtrado = df_alertas[df_alertas['Status'].isin(filtro_estado)].drop(columns=['Status'])
    
    # Renderizar tabla HTML para soportar colores y puntos de semáforo
    st.write(df_filtrado.to_html(escape=False, index=False), unsafe_allow_html=True)

# Cerrar sesión
db.close()

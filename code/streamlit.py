#---- importamos librerias
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
print("Python Executable Path (sys.executable):", sys.executable)

try:
    import nbformat
    print("nbformat version (from within Jupyter):", nbformat.__version__)
except ImportError:
    print("nbformat is not installed in this environment.")

st.set_page_config(layout="wide")

#---- CSS personalizado para el fondo de la página y los nuevos estilos
st.markdown(
    """
    <style>
    /* Fondo general de la página a blanco */
    body {
        background-color: white;
    }

    /* Estilo del título principal con fondo azul, letra blanca y sombra */
    .blue-background-white-text {
        background-color: #1c4587ff; /* Azul del tema */
        color: white; /* Letra blanca para contrastar con el azul oscuro */
        padding: 10px;
        border-radius: 0px;
        text-align: center;
        font-size: 37px;
        font-weight: bold;
        box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.4);
        margin-bottom: 60px; /* Aumentado el margen inferior para MÁS espacio */
    }

    /* Estilo para los headers de las columnas */
    .column-header {
        color: #1c4587; /* Un azul oscuro que contrasta bien con fondo blanco */
        font-size: 25px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Estilo para el separador (st.divider) */
    .st-emotion-cache-c3y6gr {
        border-top: 1px solid #666666;
        margin: 3px 0;
    }

    /* Ajuste para el texto dentro del total de clientes y saldo */
    .st-emotion-cache-nahz7x { /* Esta es una clase generada por Streamlit para st.markdown content */
        color: #1c4587; /* Asegura que el color de los números grandes sea el azul oscuro */
    }

    /* Ajuste para el texto "Total Clientes" y "Saldo Admon Total en MDP" */
    .st-emotion-cache-nahz7x > span {
        color: #666666; /* Un gris oscuro para las etiquetas */
    }

    /* Ajuste para el color de la leyenda del saldo administrado */
    .legend-saldo {
        color: #1c4587; /* Azul oscuro para el texto "Saldo Administrado por Cluster en MDP" */
    }

    /* Nuevo estilo para la etiqueta del grupo rotada y posicionada */
    .rotated-group-label-container {
        display: flex;
        align-items: center; /* Centrar verticalmente el contenido */
        justify-content: flex-end; /* Empujar el contenido a la derecha de la columna */
        height: 100%; /* Ocupa la altura completa de la columna para centrado */
        position: relative; /* Necesario para posicionar el texto rotado */
    }

    .rotated-group-label {
        font-size: 30px;
        font-weight: bold;
        color: #0492b8; /* Color del texto del grupo */
        white-space: nowrap; /* Evita que el texto se rompa */
        transform: rotate(-90deg); /* Girar 90 grados en sentido anti-horario */
        transform-origin: right center; /* Punto de origen de la rotación: derecha y centro */
        position: absolute; /* Posicionamiento absoluto dentro del contenedor flex */
        right: 0px; /* Pegar al borde derecho del contenedor (que estará pegado al gráfico) */
        padding-bottom: 0px; /* Ajuste fino si es necesario */
    }

    </style>
    """,
    unsafe_allow_html=True
)
# --- Fin del CSS personalizado ---

# Título principal con letra blanca y fondo azul
st.markdown(
    """
    <div class="blue-background-white-text">
        <h3 style="font-size: 38px; font-weight: bold; margin:0;">Estrategia Proactiva de Retención de Clientes mediante Modelado de Deserción</h3>
    </div>
    """,
    unsafe_allow_html=True
)

#---------------------------------------------------------------------------#
#---- HISTOGRAMA PARA LA PROBABILIDAD DE DESERCIÓN E INFORMACIÓN GENERAL----#
#---------------------------------------------------------------------------#

#---- leeemos los datos
file_path = '/Users/anayelicocoletzi/Documents/MCD/1_ArqDatos/trabajofinal/raw/data_stream/histigrama.csv'
base_sel = pd.DataFrame() # Inicializa base_sel con un DataFrame vacío por defecto
try:
    base_sel = pd.read_csv(file_path)
    print("Archivo leído exitosamente. Primeras 5 filas:")
except FileNotFoundError:
    print(f"Error: El archivo no se encontró en '{file_path}'")
    st.error(f"Error: El archivo '{file_path}' no se encontró. Asegúrate de que la ruta sea correcta.")
except Exception as e:
    print(f"Ocurrió un error al leer el archivo: {e}")
    st.error(f"Ocurrió un error al leer el archivo: {e}")


#---- funcion para graficar la distribución de probabilidad
def histo_proba(data_series: pd.Series, group_name: str = ""):
    """
    Crea un gráfico de distribución de probabilidad para la deserción usando KDE
    con la forma de una curva de campana.

    Args:
        data_series (pd.Series): Una serie de datos que representa las probabilidades o puntuaciones de riesgo.
        group_name (str): Nombre del grupo (ej. "Bajo Riesgo", "Mediano Riesgo").

    Returns:
        go.Figure: Un objeto Figure de Plotly que representa el gráfico de distribución.
    """
    promedio_formateado = "N/A" # Valor por defecto

    # Manejo de datos vacíos o solo NaNs para crear un gráfico de ejemplo y evitar errores
    if data_series.empty or data_series.isnull().all():
        print(f"Advertencia: La serie de datos para '{group_name}' está vacía o contiene solo NaNs. Creando gráfico de ejemplo.")
        x_kde = np.linspace(0, 1, 500)
        # Ajusta las campanas de ejemplo para que coincidan con las expectativas (26%, 50%, 90%)
        if group_name == "Bajo Riesgo":
            y_kde = np.exp(-((x_kde - 0.26)**2) / (2 * 0.05**2)) # Centrado en 0.26
            promedio_formateado = "26.0"
        elif group_name == "Mediano Riesgo":
            y_kde = np.exp(-((x_kde - 0.5)**2) / (2 * 0.05**2)) # Centrado en 0.50
            promedio_formateado = "50.0"
        elif group_name == "Alto Riesgo":
            y_kde = np.exp(-((x_kde - 0.9)**2) / (2 * 0.05**2)) # Centrado en 0.90
            promedio_formateado = "90.0"
        else:
            y_kde = np.exp(-((x_kde - 0.5)**2) / (2 * 0.1**2)) # Campana genérica
    else:
        promedio_formateado = f"{data_series.mean()*100:.1f}"
        kde = gaussian_kde(data_series.dropna())
        x_kde = np.linspace(data_series.min(), data_series.max(), 500)
        y_kde = kde(x_kde)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_kde,
            y=y_kde,
            mode='lines',
            line=dict(color='rgba(164, 194, 244, 1)', width=1),
            fill='tozeroy',
            fillcolor='rgba(164, 194, 244, 0.5)',
            showlegend=False))

    fig.add_annotation(xref="paper", yref="paper",
                       x=0.5, y=0.02,            
                       #text=f"<b><span style='font-size: 45px;'>{promedio_formateado}%</span></b><br><span style='font-size: 25px;'>Prob. de deserción</span>",
                       text=f"<b><span style='font-size: 50px;'>{promedio_formateado}%</span></b> <span style='font-size: 20px;'> Prob. de deserción</span>", # <--- CAMBIO CLAVE AQUÍ
                       showarrow=False,
                       font=dict(color="#1155cc"),
                       align="center",          
                       xanchor="center",          
                       yanchor="bottom",
                       bgcolor="rgba(255,255,255,0)",
                       bordercolor="rgba(0,0,0,0)",
                       borderwidth=0
                      )
    
    fig.update_layout(plot_bgcolor='white',
                      paper_bgcolor='white',
                      title=None, # Asegúrate de que no haya un título principal de Plotly configurado aquí
                      height=200, # Altura de la caja del gráfico Plotly (ajusta si es necesario)
                      width=250,  # Ancho de la caja del gráfico Plotly (se puede ajustar con use_container_width=True)
                      xaxis=dict(title=None,
                                 showticklabels=False,
                                 showgrid=False,
                                 zeroline=False),
                      yaxis=dict(title=None,
                                 showticklabels=False,
                                 showgrid=False,
                                 zeroline=False),
                      legend=dict(bgcolor='white',
                                  font=dict(color='black')),
                      margin=dict(t=0, b=16, l=16, r=0)
                     )
    return fig # Solo devolvemos la figura

#---- Crear los gráficos de distribución de probabilidad para cada grupo
hist1 = None
hist2 = None
hist3 = None

if not base_sel.empty and 'grupo' in base_sel.columns and 'probabilidad' in base_sel.columns:
    if 1 in base_sel['grupo'].unique():
        hist1 = histo_proba(data_series=base_sel[base_sel['grupo']==1]['probabilidad'], group_name="Bajo Riesgo")
    if 2 in base_sel['grupo'].unique():
        hist2 = histo_proba(data_series=base_sel[base_sel['grupo']==2]['probabilidad'], group_name="Mediano Riesgo")
    if 3 in base_sel['grupo'].unique():
        hist3 = histo_proba(data_series=base_sel[base_sel['grupo']==3]['probabilidad'], group_name="Alto Riesgo")
else:
    st.warning("No se pudieron cargar los datos o faltan columnas 'grupo' o 'probabilidad' en el archivo.")
    # Crea gráficos de ejemplo si los datos no se cargaron correctamente
    hist1 = histo_proba(pd.Series(), group_name="Bajo Riesgo") # Pasamos una serie vacía para que use los valores predefinidos
    hist2 = histo_proba(pd.Series(), group_name="Mediano Riesgo")
    hist3 = histo_proba(pd.Series(), group_name="Alto Riesgo")


# --- Definición de las columnas ---
col1, col2, col3, col4 = st.columns([2, 2, 2, 2.3])

with col1:
    st.markdown("<h3 class='column-header' style='color: #595959; font-weight: bold; font-size: 30px;'>Bajo Riesgo</h3>", unsafe_allow_html=True)
    if hist1: # Solo muestra si el gráfico se pudo crear
        st.plotly_chart(hist1, use_container_width=True)

with col2:
    st.markdown("<h3 class='column-header' style='color: #595959; font-weight: bold; font-size: 30px;'>Mediano Riesgo</h3>", unsafe_allow_html=True)
    if hist2: # Solo muestra si el gráfico se pudo crear
        st.plotly_chart(hist2, use_container_width=True)

with col3:
    st.markdown("<h3 class='column-header' style='color: #595959; font-weight: bold; font-size: 30px;'>Alto Riesgo</h3>", unsafe_allow_html=True)
    if hist3: # Solo muestra si el gráfico se pudo crear
        st.plotly_chart(hist3, use_container_width=True)

with col4:
    # Calculate the total number of clients
    total_clientes = 0
    if not base_sel.empty:
        total_clientes = len(base_sel) 
        suma_saldo = (base_sel['saldopararetiro'].sum()/1000000).round(1)

    # Display the client count with large font aligned to the left
    st.markdown(f"""
                    <div style="border-left: 3px solid #b7b7b7; text-align: left; margin-top: 25px; padding-left: 25px;">
                    <span style="font-size: 50px; font-weight: bold; color: #1c4587;">{total_clientes:,}</span>
                    <br>
                    <span style="font-size: 25px; font-weight: bold; color: #595959;">Total Clientes</span>
                    <br>
                    <span style="font-size: 50px; font-weight: bold; color: #1c4587;">${suma_saldo:,.1f}</span>
                    <br>
                    <span style="font-size: 25px; font-weight: bold; color: #595959;">Saldo Admon Total en MDP</span>
                </div>
            """, unsafe_allow_html=True)
    
#---------------------------------------------------------------------------#
#---- GRÁFICOS DE SALDO ADMNISTRADO POR GRUPOS (SEGMNETACIPON ANTERIOR) ----#
#---------------------------------------------------------------------------#

#---- agregamos la leyenda del saldo
st.markdown(f"""
    <div style="display: flex; align-items: center; margin-top: 40px; margin-bottom: 20px;">
        <hr style="flex-grow: 1.5; border-top: 2px solid #0492b8; margin-right: 10px;">
        <div style="text-align: center;">
            <span class="legend-saldo" style="font-size: 35px; font-weight: bold;">Saldo Administrado por Cluster en MDP</span>
        </div>
        <hr style="flex-grow: 1.5; border-top: 2px solid #0492b8; margin-left: 10px;">
    </div>
""", unsafe_allow_html=True)

#---- leeemos los datos
file_path = '/Users/anayelicocoletzi/Documents/MCD/1_ArqDatos/trabajofinal/raw/data_stream/base_imp.csv'
base_imp = pd.DataFrame() # Inicializa base_sel con un DataFrame vacío por defecto
try:
    base_imp = pd.read_csv(file_path)
    print("Archivo leído exitosamente. Primeras 5 filas:")
except FileNotFoundError:
    print(f"Error: El archivo no se encontró en '{file_path}'")
    st.error(f"Error: El archivo '{file_path}' no se encontró. Asegúrate de que la ruta sea correcta.")
except Exception as e:
    print(f"Ocurrió un error al leer el archivo: {e}")
    st.error(f"Ocurrió un error al leer el archivo: {e}")

#---- inicializamos las variables
# Asegúrate de que estas variables existan antes de usarlas
saldo_prom1 = base_imp[base_imp['grupo'] == 1]['saldo_pro'].iloc[0] if 1 in base_imp['grupo'].unique() else 0
saldo_prom2 = base_imp[base_imp['grupo'] == 2]['saldo_pro'].iloc[0] if 2 in base_imp['grupo'].unique() else 0
saldo_prom3 = base_imp[base_imp['grupo'] == 3]['saldo_pro'].iloc[0] if 3 in base_imp['grupo'].unique() else 0


#---- creamos la función para graficar el saldo administrado
def pie_saldo(data: pd.DataFrame, grupo: int, color: str):
    """
    Crea un gráfico de dona para el saldo administrado por grupo.

    Args:
        data (pd.DataFrame): DataFrame con el saldo por grupo.
        grupo (int): El número de grupo a destacar.
        color (str): El color para la rebanada del grupo destacado.

    Returns:
        go.Figure: Un objeto Figure de Plotly que representa el gráfico de dona.
    """
    if data.empty or grupo not in data['grupo'].values:
        print(f"Advertencia: Datos insuficientes para el grupo {grupo}. Creando gráfico de dona de ejemplo.")
        valor = 100.0 # Example value
        valor_tot = 300.0 # Example total
        datos_dona = pd.DataFrame({
            'grupo': ['Grupo', 'Resto_Ctes'],
            'saldo': [valor, valor_tot - valor]
        })
    else:
        #---- datos para la dona
        valor_tot = base_imp['saldo_tot_mdp'].sum()
        valor = base_imp[base_imp['grupo'] == grupo]['saldo_tot_mdp'].iloc[0]
        valor_res = valor_tot - valor

        datos_dona = pd.DataFrame({
            'grupo': [f'Ctes. Focalizados', 'Resto Ctes'],
            'saldo': [valor, valor_res]
        })

    #---- creamos la gráfica de dona
    fig = go.Figure(data=[go.Pie(
        labels=datos_dona['grupo'],
        values=datos_dona['saldo'],
        hole=0.5,
        textinfo='none', # Esto asegura que no haya texto en las rebanadas
        hoverinfo='label+percent', # Muestra etiqueta, valor y porcentaje al pasar el mouse
        marker=dict(
            colors=[color, 'lightgray']
        ),
        insidetextorientation='radial',
        direction='counterclockwise',
    )])

    # Add the central annotation
    fig.add_annotation(
        text=f"<b><span style='font-size: 50px; color:#1c4587;'>${valor:,.1f}</span></b>",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(color='black'),
        align="center"
    )

    # Make the center of the donut white
    fig.update_layout(
        plot_bgcolor='white', # Set plot background to white
        paper_bgcolor='white', # Set paper background to white
        title_text=None,
        showlegend=False,
        height=300,
        width=350,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig # IMPORTANT: Return the figure

#---- Crear los gráficos de acumulación de saldo
pie1 = None
pie2 = None
pie3 = None

if not base_imp.empty: # Check if saldo_grupo has data
    if 1 in base_imp['grupo'].unique():
        pie1 = pie_saldo(data=base_imp, grupo=1, color='rgb(224,102,102)') # Reddish for low risk
    if 2 in base_imp['grupo'].unique():
        pie2 = pie_saldo(data=base_imp, grupo=2, color='rgb(255,207,54)')  # Yellowish for medium risk
    if 3 in base_imp['grupo'].unique():
        pie3 = pie_saldo(data=base_imp, grupo=3, color='rgb(135,197,68)') # Greenish for high risk
else:
    st.warning("No se pudo generar los gráficos de saldo. Los datos de 'saldo_grupo' no están disponibles.")
    # Create example charts if saldo_grupo is empty
    pie1 = pie_saldo(pd.DataFrame({'grupo': [1], 'saldo_mdp': [100]}), grupo=1, color='rgb(224,102,102)')
    pie2 = pie_saldo(pd.DataFrame({'grupo': [2], 'saldo_mdp': [100]}), grupo=2, color='rgb(255,207,54)')
    pie3 = pie_saldo(pd.DataFrame({'grupo': [3], 'saldo_mdp': [100]}), grupo=3, color='rgb(135,197,68)')


#---- definición de las columnas para los gráficos de dona
# Damos menos espacio a la columna de la etiqueta, y la posicionamos de forma absoluta dentro
col_group1_label, col_pie1, col_group2_label, col_pie2, col_group3_label, col_pie3 = st.columns([0.1, 1, 0.1, 1, 0.1, 1])

with col_group1_label:
    st.markdown("""
        <div class='rotated-group-label-container'>
            <span class='rotated-group-label'>Bajo Riesgo</span>
        </div>
    """, unsafe_allow_html=True)
with col_pie1:
    if pie1:
        st.plotly_chart(pie1, use_container_width=True)
    st.markdown(f"<h3 class='column-header' style='margin-top: -10px;font-size: 25px;font-weight: bold; color: #595959;'>${saldo_prom1:,.0f} promedio por cte.</h3>", unsafe_allow_html=True)

with col_group2_label:
    st.markdown("""
        <div class='rotated-group-label-container'>
            <span class='rotated-group-label'>Mediano Riesgo</span>
        </div>
    """, unsafe_allow_html=True)
with col_pie2:
    if pie2:
        st.plotly_chart(pie2, use_container_width=True)
    st.markdown(f"<h3 class='column-header' style='margin-top: -10px;font-size: 25px;font-weight: bold; color: #595959;'>${saldo_prom2:,.0f} promedio por cte.</h3>", unsafe_allow_html=True)

with col_group3_label:
    st.markdown("""
        <div class='rotated-group-label-container'>
            <span class='rotated-group-label'>Alto Riesgo</span>
        </div>
    """, unsafe_allow_html=True)
with col_pie3:
    if pie3:
        st.plotly_chart(pie3, use_container_width=True)
    st.markdown(f"<h3 class='column-header' style='margin-top: -10px;font-size: 25px;font-weight: bold; color: #595959;'>${saldo_prom3:,.0f} promedio por cte.</h3>", unsafe_allow_html=True)

#---------------------------------------------------------------------------#
#---- GRÁFICO DE DISTRIBUCIÓN POR RANGO DE EDAD Y GÉNERO POR GRUPO ANTE ----#
#---------------------------------------------------------------------------#

# ---- Agregamos un divisor para la sección de edad y género ----
st.markdown(f"""
    <div style="display: flex; align-items: center; margin-top: 60px; margin-bottom: 20px;">
        <hr style="flex-grow: 1.5; border-top: 2px solid #0492b8; margin-right: 10px;">
        <div style="text-align: center;">
            <span class="legend-saldo" style="font-size: 35px; font-weight: bold;">Distribución por Rango de Edad y Género por Grupo</span>
        </div>
        <hr style="flex-grow: 1.5; border-top: 2px solid #0492b8; margin-left: 10px;">
    </div>
""", unsafe_allow_html=True)

#---- cargamos la tabla que creamos:
data_edad = pd.read_csv('/Users/anayelicocoletzi/Documents/MCD/1_ArqDatos/trabajofinal/raw/data_stream/dis_edad.csv')

# Convert 'sexo' to string type to handle mixed types gracefully
data_edad['sexo'] = data_edad['sexo'].astype(str)

# Map known values and treat anything else as NaN for consistent dropping
data_edad['sexo'] = data_edad['sexo'].replace({
    'Hombre': 'Hombre',
    'Mujer': 'Mujer'
})

# Explicitly convert remaining non-Hombre/Mujer values to NaN (e.g., '', 'nan', 'null')
data_edad['sexo'] = data_edad['sexo'].apply(lambda x: x if x in ['Hombre', 'Mujer'] else pd.NA)

# Drop rows where 'sexo' is truly NaN (including those just converted to NaN)
data_edad = data_edad.dropna(subset=['sexo'])

if not base_sel.empty and 'grupo' in base_sel.columns:
    grupo_mapping = {1: 'Bajo Riesgo', 2: 'Mediano Riesgo', 3: 'Alto Riesgo'}
    data_edad['grupo_display'] = data_edad['grupo'].replace(grupo_mapping)

    data_edad = data_edad[data_edad['grupo_display'].isin(list(grupo_mapping.values()))].copy()
else:
    st.warning("`base_sel` es nulo o no contiene la columna 'grupo'. Los títulos de los subplots podrían no ser correctos.")
    data_edad['grupo_display'] = data_edad['grupo'].astype(str) # Fallback to string original numbers


#---- definir rangos de edad 
age_bins = [0, 25, 30, 35, 40, 45, 50, 55, 60, 100]
age_labels = ['0-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60+']

data_edad['rango_edad'] = pd.Categorical(data_edad['rango_edad'], categories=age_labels, ordered=True)
data_edad = data_edad.sort_values('rango_edad')

# Definir colores personalizados para el género
gender_colors = {
    'Hombre': '#275ba6', # Color para hombres
    'Mujer': '#36b9e6'    # Color para mujeres
}

# Configurar el gráfico con Plotly Express
fig = px.bar(
    data_edad,
    x='rango_edad',               # Rangos de edad en el eje X
    y='frecuencia_bidireccional', # Frecuencia bidireccional en el eje Y
    color='sexo',               # Colorear por género
    facet_col='grupo_display',  # AHORA USA LA NUEVA COLUMNA DE DISPLAY PARA FACETAS
    barmode='relative',           # Esencial para que las barras negativas y positivas se extiendan desde el cero
    orientation='v',              # Barras verticales
    labels={
        'rango_edad': 'Rango de Edad',
        'frecuencia_bidireccional': 'Frecuencia', # Este label ya no se mostrará como título
        'sexo': 'Género'
    },
    height=500, # Altura de la figura
    width=350 * len(data_edad['grupo_display'].unique()), # Ancho dinámico basado en el número de clusters
    category_orders={"rango_edad": age_labels}, # Asegurar el orden de las edades en el eje X
    color_discrete_map=gender_colors,
    custom_data=['frecuencia', 'porcentaje', 'total_genero'],
    facet_col_spacing=0.08 # Ajusta este valor para más o menos espacio. Por ejemplo, 0.08, 0.1, 0.15
)

# Iteramos sobre cada trace (barra de Hombres y Mujeres)
for trace in fig.data:
    if trace.name == 'Hombre':
        trace.hovertemplate = (
            "<b>Hombres:</b> %{customdata[0]}<br>" # Frecuencia absoluta (customdata[0] es 'frecuencia')
            "<b>Porcentaje de Hombres en el cluster:</b> %{customdata[1]:,.2f}%<br>" # customdata[1] es 'porcentaje'
            "<b>Total Hombres en cluster:</b> %{customdata[2]}<extra></extra>" # customdata[2] es 'total_genero'
        )
    elif trace.name == 'Mujer':
        trace.hovertemplate = (
            "<b>Mujeres:</b> %{customdata[0]}<br>"
            "<b>Porcentaje de Mujeres en el cluster:</b> %{customdata[1]:,.2f}%<br>"
            "<b>Total Mujeres en cluster:</b> %{customdata[2]}<extra></extra>"
        )

# Ajustes del layout general de la figura
fig.update_layout(
    title={'text': '',
        'y':0.98,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(
            size=1,
            color='#1c4587'
        )
    },
    bargap=0.1,  # Espacio entre grupos de barras (dentro de una sola faceta)
    plot_bgcolor='white', # Fondo del área del gráfico
    paper_bgcolor='white', # Fondo de la figura
    legend_title_text='Género',
    legend=dict(
                font=dict(size=14,color='#595959'),
                # ******* POSICIÓN DE LA LEYENDA ABAJO *******
                orientation="h",  # Horizontal
                yanchor="top",    # Anclar a la parte superior de su contenedor
                y=1,           # Mover la leyenda hacia abajo (ajusta este valor si necesitas más/menos espacio)
                xanchor="center", # Anclar al centro horizontal
                x=0.5           # Centrar horizontalmente
            )
)

# Ajustar el rango del eje Y para que sea simétrico en todos los subplots
max_freq_global = data_edad['frecuencia'].max()

# Create a mapping from original group numbers to display names for Y-axis titles
grupo_axis_titles = {
    1: 'Bajo Riesgo',
    2: 'Mediano Riesgo',
    3: 'Alto Riesgo'
}

# --- IMPORTANT CHANGES HERE ---
# 1. Hide original facet titles
fig.for_each_annotation(lambda a: a.update(text='', visible=False)) # Hide the default facet titles

# 2. Iterate through each subplot's Y-axis to set its title
for i, group_id in enumerate(sorted(data_edad['grupo'].unique())): # Sort to ensure consistent order
    yaxis_key = f'yaxis{i+1}' if i > 0 else 'yaxis'
    group_name = grupo_axis_titles.get(group_id, f'Grupo {group_id}')

    fig.update_layout(
        **{yaxis_key: dict(
            visible=True,
            showticklabels=False,
            title={
                'text': group_name,
                'font': dict(size=20, color='rgb(106,168,79)', weight='bold'), # Asegúrate de que el tamaño y color de la leyenda estén aquí
                'standoff': 5 # Distancia entre el título del eje Y y los datos
            },
            range=[-max_freq_global * 1.1, max_freq_global * 1.1],
            tickvals=[-max_freq_global, -int(max_freq_global/2), 0, int(max_freq_global/2), max_freq_global],
            ticktext=[str(max_freq_global), str(int(max_freq_global/2)), '0', str(int(max_freq_global/2)), str(max_freq_global)],
            automargin=True
        )}
    )

st.plotly_chart(fig, use_container_width=True)

#---------------------------------------------------------------------------#
#---- SELECCIÓN PARTICULAR DE GRUPO Y ESTRATEGIA DE RETENCIÓN A LANZAR  ----#
#---------------------------------------------------------------------------#

# ---- Agregamos un divisor para la sección de edad y género ----
st.markdown(f"""
    <div style="display: flex; align-items: left; margin-top: 60px; margin-bottom: 20px;">
        <div style="text-align: left;">
            <span class="legend-saldo" style="font-size: 35px; font-weight: bold;">Estrategias de Retención</span>
        </div>
    </div>
""", unsafe_allow_html=True)


def run():

    st.markdown(
    """
    <style>
    /* Clase base para el tamaño de todos los cuadros */
    .common-box-sizing {
        width: 280px; /* Ancho fijo para todos los cuadros */
        min-height: 250px; /* Altura mínima para todos los cuadros */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    /* Estilo para los cuadros 3D (Columnas 2, 3, 4) */
    .card-3d {
        border: 2px solid #ddd; /* Borde sutil */
        border-radius: 25px; /* Esquinas redondeadas */
        padding: 10px;
        background-color: #f9f9f9; /* Fondo claro */
        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.5), /* Sombra principal para efecto 3D */
                    -5px -5px 15px rgba(255, 255, 255, 0.7); /* Sombra de luz para realce */
        transition: transform 0.3s ease-in-out; /* Transición suave al pasar el ratón */
    }
    .card-3d:hover {
        transform: translateY(-5px); /* Pequeño levantamiento al pasar el ratón */
    }

    /* Estilo específico para la primera columna si no es 3D pero queremos el mismo tamaño */
    /* .column-1-box {*/
        /*border: 1px solid #ffffffff;*/
        /*padding: 1px;*/
        /*border-radius: 5px;*/
    /*}*/

    /* Asegura que los checkboxes sean más grandes */
    div[class="st-bf"] label {
        font-size: 30px !important;
        margin-left: 20px;
    }

    /* Estilo para el contenedor del checkbox debajo de las tarjetas */
    .checkbox-container {
        margin-top: 25px; /* Ajusta este valor para mover el checkbox hacia abajo */
        text-align: center; /* Centra el checkbox debajo de la tarjeta */
    }

    /* --- ESTILOS AJUSTADOS PARA LA PRIMERA COLUMNA --- */
    .clientes-objetivo-content {
        padding-left: 20px;
        padding-top: 0px; /* Mantiene el espacio superior dentro de la caja */
    }
    .clientes-objetivo-content .main-title {
        font-size: 30px;
        font-weight: bold;
        color: #1c4587;
        margin-bottom: 0px; /* Reduce el espacio debajo de "Clientes Objetivo" */
        display: block; /* Hace que el span se comporte como un bloque para que margin-bottom funcione */
    }
    .clientes-objetivo-content .subtitle {
        font-size: 18px;
        font-weight: bold;
        color: #595959;
        margin-top: 0px; /* Elimina el espacio superior de "Selecciona una opción" */
        margin-bottom: 0px; /* Ajusta el espacio debajo de "Selecciona una opción" y antes de los checkboxes */
        display: block; /* Hace que el span se comporte como un bloque */
    }

    /* Estilo para el div que contiene los checkboxes de riesgo */
    .riesgo-checkboxes-container {
        /* Eliminamos padding-left y usamos text-align o flexbox para centrar */
        text-align: center; /* Centra el contenido inline-block como los checkboxes */
    }

    </style>
    """,
    unsafe_allow_html=True
    )
    
    # Crear 4 columnas para la sección superior
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    # Initialize session state for checkboxes in col1
    if 'selected_riesgo' not in st.session_state:
        st.session_state.selected_riesgo = None

    def handle_riesgo_selection(riesgo):
        if st.session_state.selected_riesgo == riesgo:
            st.session_state.selected_riesgo = None  # Deselect if already selected
        else:
            st.session_state.selected_riesgo = riesgo

    with col1:
        st.markdown(
            f"""
            <div class="column-1-box">
                <div style="text-align: left; font-size: 50px; color: #008aaf; font-weight: bold; margin-bottom: 0px;
                margin-left: 20px; margin-top: 0px;">
                 <span style="font-size: 30px; font-weight: bold; color: #595959;margin-bottom: 0px;">Clientes Objetivo</span>
                <br>
                <span style="font-size: 20px; font-weight: bold; margin-top: 25px; color: #595959;">Selecciona un opción</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        #opciones_texto = ["Bajo Riesgo", "Mediano Riesgo", "Alto Riesgo"]
        
        # Ahora, renderizar los checkboxes de Streamlit directamente después del markdown de la caja.
        # Envolvemos los checkboxes individuales en su propio div para aplicar el centrado.
        st.markdown('<div class="riesgo-checkboxes-container">', unsafe_allow_html=True)
        opciones_texto = ["Bajo Riesgo", "Mediano Riesgo", "Alto Riesgo"]
        for riesgo in opciones_texto:
            st.checkbox(
                riesgo,
                key=f"chk_{riesgo}",
                value=(st.session_state.selected_riesgo == riesgo),
                on_change=handle_riesgo_selection,
                args=(riesgo,)
            )
        st.markdown('</div>', unsafe_allow_html=True) # Cierra el div del contenedor

        opcion_col1 = st.session_state.selected_riesgo

        
    # Lógica para manejar la selección única de imágenes (Col 2, 3, 4 - unchanged)
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None

    def handle_image_selection(image_name):
        # Deseleccionar si ya está seleccionada o seleccionar si no lo está
        if st.session_state.selected_image == image_name:
            st.session_state.selected_image = None
        else:
            st.session_state.selected_image = image_name

    # Dictionary to store checkbox states (Col 2, 3, 4 - unchanged)
    checkbox_states = {
        "chk_img1": False,
        "chk_img2": False,
        "chk_img3": False,
    }
    
    with col2:
        st.markdown(
            f"""
            <div class="card-3d">
                <div style="text-align: left; font-size: 60px; color: #008aaf; font-weight: bold; margin-bottom: 0px;
                margin-left: 20px; margin-top: 0px;">
                    01
                </div>
                <div style="display: flex; justify-content: center; font-size: 4em; top: 0px;">
                    💡
                </div>
                <div style="text-align: center; font-size: 20px; color: #1c4587; font-weight: bold; margin-bottom: 0 px;
                margin-left: 10px; margin-right: 10px;">
                    Asesoría Financiera Personalizada
                </div>
                
            </div>
            """,
            unsafe_allow_html=True
        )
        checkbox_states["chk_img1"] = st.checkbox("Estrategia 1", key="chk_img1",
                                   value=(st.session_state.selected_image == "Imagen 1"),
                                   on_change=handle_image_selection, args=("Imagen 1",))

    with col3: 
        st.markdown(
            f"""
            <div class="card-3d">
                <div style="text-align: left; font-size: 60px; color: #008aaf; font-weight: bold; margin-bottom: 0px;
                margin-left: 20px; margin-top: 0px;">
                    02
                </div>
              <div style="display: flex; justify-content: center; font-size: 4em; top: 0px;">
                    🤝
                </div>
                <div style="text-align: center; font-size: 20px; color: #1c4587; font-weight: bold; margin-bottom: 0px;
                margin-left: 10px; margin-right: 10px;">
                    Comunicación Segmentada y Relevante
                </div>
                
            </div>
            """,
            unsafe_allow_html=True
        )
        checkbox_states["chk_img2"] = st.checkbox("Estrategia 2", key="chk_img2",
                                   value=(st.session_state.selected_image == "Imagen 2"),
                                   on_change=handle_image_selection, args=("Imagen 2",))

    with col4:
        st.markdown(
            f"""
            <div class="card-3d">
                <div style="text-align: left; font-size: 60px; color: #008aaf; font-weight: bold; margin-bottom: 0px;
                margin-left: 20px; margin-top: 0px;">
                    03
                </div>
                <div style="display: flex; justify-content: center; font-size: 4em; top: 0px;">
                    🌟
                </div>
                <div style="text-align: center; font-size: 20px; color: #1c4587; font-weight: bold; margin-bottom: 0px;
                margin-left: 10px; margin-right: 10px;">
                    Beneficios
                </div>
                <div style="text-align: center; font-size: 20px; color: #1c4587; font-weight: bold; margin-bottom: 0px;
                margin-left: 10px; margin-right: 10px;">
                    Exclusivos
                </div>                
            </div>
            """,
            unsafe_allow_html=True
        )
        checkbox_states["chk_img3"] = st.checkbox("Estrategia 3", key="chk_img3",
                                   value=(st.session_state.selected_image == "Imagen 3"),
                                   on_change=handle_image_selection, args=("Imagen 3",))

    
    opcion_columnas_restantes = st.session_state.selected_image

    selected_images_text = []
    for key, selected in checkbox_states.items():
        if selected:
            image_number = key.replace("chk_img", "")  # Extract image number
            selected_images_text.append(f"Estrategia {image_number}")

    if opcion_col1 and selected_images_text:
        st.success(f"Nos enfocaremos en los clientes de **{opcion_col1}** y aplicaremos **{', '.join(selected_images_text)}**.")
    elif opcion_col1:
        st.warning("Te falta selecionar una estrategia.")
        st.info(f"Nos enfocaremos en los clientes de **{opcion_col1}**.")
    elif selected_images_text:
        st.warning("Selecciona los clientes a quienes deseas enfocar.")
        st.info(f"Aplicaremos la **{', '.join(selected_images_text)}**.")
    else:
        st.error("Por favor, selecciona un grupo de clientes y la estrategia que deseas aplicar.")

    #--- Nueva Fila Condicional para el contenido y el gráfico ---
    if opcion_col1 and selected_images_text:
        col_texto, col_grafico = st.columns([1, 1])

        communication_matrix = {
            "Bajo Riesgo": {
                "Estrategia 1": """
                    <div style="text-align: justify; font-size: 16px; color: black;width: 650px;border-right: 3px solid #b7b7b7 ; padding-right: 50px;">
                        <h3><span style="color: #38761d;font-weight: bold;">Comunicación para Clientes de Bajo Riesgo con Estrategia 1</span></h3>
                        <p><span style="color: #1155cc; font-weight: bold;">Objetivo:</span> Fomentar la estabilidad financiera, educar sobre el ahorro y presentar opciones de bajo riesgo.</p>
                        <p><span style="color: #1155cc; font-weight: bold;">Canales y Frecuencia:</span></p>
                        <ul>
                            <li>Email Marketing (Quincenal): Boletines informativos con un resumen de consejos de ahorro, "tips del mes" para reducir gastos, y comparativas sencillas de productos de inversión conservadores (ej. fondos de inversión de deuda, CETES).</li>
                            <li>Notificaciones Push en App (Semanal): Alertas sobre el rendimiento de sus productos de ahorro, recordatorios para revisar sus presupuestos o nuevas funcionalidades de la app enfocadas en control de gastos.</li>
                        </ul>
                        <p><span style="color: #1155cc; font-weight: bold;">Contenido Específico:</span></p>
                        <ul>
                            <li>"Tu guía para iniciar a invertir: Lo que necesitas saber sobre fondos de bajo riesgo."</li>
                            <li>"Ahorra para tu próxima meta: Estrategias para alcanzar tus objetivos."</li>
                            <li>"¡Novedad! La herramienta de presupuesto inteligente que estabas esperando."</li>
                        </ul>
                    </div>
                """,
                "Estrategia 2": """
                    <div style="text-align: justify; font-size: 16px; color: black;width: 650px;border-right: 3px solid #b7b7b7 ; padding-right: 50px;">
                        <h3><span style="color: #38761d;font-weight: bold;">Comunicación para Clientes de Bajo Riesgo con Estrategia 2</span></h3>
                        <p><span style="color: #1155cc; font-weight: bold;">Enfoque:</span> Construcción de bases financieras, educación sobre productos básicos y fomento del ahorro.</p>
                        <p><span style="color: #1155cc; font-weight: bold;">Acciones Concretas:</span></p>
                        <ul>
                            <li>"Taller de Presupuesto Inteligente" (Online, Mensual): Sesiones interactivas en vivo para enseñar a crear y mantener un presupuesto, identificar fugas de dinero y establecer metas de ahorro.</li>
                            <li>Línea Directa "Pregúntale a tu Asesor" (Horario Fijo): Un canal dedicado en la app o web donde pueden hacer preguntas básicas sobre ahorro, deudas o productos financieros con respuestas rápidas de asesores junior.</li>
                            <li>Guías Interactivas: Contenido descargable o web sobre "Cómo iniciar tu fondo de emergencia" o "Entendiendo las tasas de interés de tus ahorros."</li>
                        </ul>
                    </div>
                 """,
                "Estrategia 3": """
                    <div style="text-align: justify; font-size: 16px; color: black;width: 650px;border-right: 3px solid #b7b7b7 ; padding-right: 50px;">
                        <h3><span style="color: #38761d;font-weight: bold;">Comunicación para Clientes de Bajo Riesgo con Estrategia 3</span></h3>
                        <p><span style="color: #1155cc; font-weight: bold;">Ventajas Concretas:</span></p>
                        <ul>
                            <li>"Tasa Plus en Ahorro": Un incremento del 0.25% sobre la tasa de interés estándar en sus cuentas de ahorro por mantener un saldo promedio.</li>
                            <li>Acceso Anticipado a Nuevas Funcionalidades: Ser los primeros en probar y dar feedback sobre nuevas herramientas en la app (ej. un simulador de metas de ahorro avanzado, una tarjeta de débito con beneficios por compras).</li>
                            <li>Descuentos en Cursos de Educación Financiera Básica: Alianzas con plataformas educativas para ofrecer cupones del 20-30% en cursos online sobre finanzas personales, manejo de crédito o inversión inicial.</li>
                        </ul>
                    </div>
                """,
            },
            "Mediano Riesgo": {
                "Estrategia 1": """
                    <div style="text-align: justify; font-size: 16px; color: black;width: 650px;border-right: 3px solid #b7b7b7 ; padding-right: 50px;">
                        <h3><span style="color: #38761d;font-weight: bold;">Comunicación para Clientes de Mediano Riesgo con Estrategia 1</span></h3>
                        <p><span style="color: #1155cc; font-weight: bold;">Objetivo:</span> Impulsar la diversificación, optimizar sus portafolios y educar sobre oportunidades con riesgo moderado.</p>
                        <p><span style="color: #1155cc; font-weight: bold;">Canales y Frecuencia:</span></p>
                        <ul>
                            <li>Webinars Educativos (Mensual): Sesiones en vivo con expertos donde se analizan tendencias de mercado, se explican estrategias de diversificación y se presentan oportunidades de inversión en sectores estables pero con potencial de crecimiento (ej. tecnología madura, bienes raíces a través de FIBRAS).</li>
                            <li>Llamadas Proactivas (Trimestral): Asesores llamando para revisar el portafolio, presentar nuevas alternativas que se alineen con su perfil de riesgo y responder dudas específicas.</li>
                        </ul>
                        <p><span style="color: #1155cc; font-weight: bold;">Contenido Específico:</span></p>
                        <ul>
                            <li>"Diversifica tu portafolio: Explorando acciones de crecimiento y bonos corporativos."</li>
                            <li>"Análisis de Mercado: ¿Dónde están las oportunidades en el próximo trimestre?"</li>
                            <li>"Invitación exclusiva: Cómo usar la volatilidad a tu favor en inversiones de mediano riesgo."</li>
                        </ul>
                    </div>
                """,
                "Estrategia 2": """
                    <div style="text-align: justify; font-size: 16px; color: black;width: 650px;border-right: 3px solid #b7b7b7 ; padding-right: 50px;">
                        <h3><span style="color: #38761d;font-weight: bold;">Comunicación para Clientes de Mediano Riesgo con Estrategia 2</span></h3>
                        <p><span style="color: #1155cc; font-weight: bold;">Enfoque:</span> Optimización del portafolio, planificación de metas a mediano plazo y gestión de riesgo.</p>
                        <p><span style="color: #1155cc; font-weight: bold;">Acciones Concretas:</span></p>
                        <ul>
                            <li>Análisis de Portafolio Anual Personalizado: Un informe detallado enviado por su asesor dedicado, revisando el rendimiento de sus inversiones, sugiriendo rebalanceos y proponiendo nuevas oportunidades de acuerdo a su perfil de riesgo.</li>
                            <li>"Sesiones de Planificación de Metas" (Por Cita): Asesoría uno a uno para planificar metas específicas como la compra de una casa, la educación de los hijos o un plan de jubilación a mediano plazo, con escenarios y proyecciones.</li>
                            <li>Talleres Temáticos sobre Inversión (Trimestral): Profundizando en temas como "Inversión en Bienes Raíces: FIBRAS y Crowdfunding" o "Fondos Indexados vs. Fondos Activos."</li>
                        </ul>
                    </div>
                """,
                "Estrategia 3": """
                    <div style="text-align: justify; font-size: 16px; color: black;width: 650px;border-right: 3px solid #b7b7b7 ; padding-right: 50px;">
                        <h3><span style="color: #38761d;font-weight: bold;">Comunicación para Clientes de Mediano Riesgo con Estrategia 3</span></h3>
                        <p><span style="color: #1155cc; font-weight: bold;">Ventajas Concretas:</span></p>
                        <ul>
                            <li>Reducción en Comisiones de Fondos de Inversión: Descuentos del 10-15% en las comisiones de administración de fondos de inversión de riesgo moderado.</li>
                            <li>Invitaciones a Eventos de Networking Exclusivos: Acceso a reuniones con otros inversionistas, charlas con líderes empresariales o eventos culturales patrocinados por la institución.</li>
                            <li>Suscripción Gratuita a Reportes de Investigación de Mercado Premium: Acceso a análisis detallados de sectores económicos específicos o tendencias de inversión globales de proveedores externos.</li>
                        </ul>
                    </div>
                """,
            },
            "Alto Riesgo": {
                "Estrategia 1": """
                    <div style="text-align: justify; font-size: 16px; color: black;width: 650px;border-right: 3px solid #b7b7b7 ; padding-right: 50px;">
                        <h3><span style="color: #38761d;font-weight: bold;">Comunicación para Clientes de Alto Riesgo con Estrategia 1</span></h3>
                        <p><span style="color: #1155cc; font-weight: bold;">Objetivo:</span> Ofrecer oportunidades de alto rendimiento, estrategias de inversión complejas y análisis de mercado profundo.</p>
                        <p><span style="color: #1155cc; font-weight: bold;">Canales y Frecuencia:</span></p>
                        <ul>
                            <li>Reuniones Personalizadas (Mensual/Bimensual): Presenciales o por videollamada con un asesor senior. Se discuten estrategias avanzadas, análisis de mercados emergentes, opciones de inversión privada y gestión de riesgo sofisticada.</li>
                            <li>Reportes de Mercado Exclusivos (Semanal): Documentos detallados enviados por email con análisis técnico y fundamental de activos de alto riesgo, proyecciones de mercado y posibles escenarios.</li>
                        </ul>
                        <p><span style="color: #1155cc; font-weight: bold;">Contenido Específico:</span></p>
                        <ul>
                            <li>"Análisis Cuantitativo: Oportunidades de inversión en criptomonedas y tecnología disruptiva."</li>
                            <li>"Estrategias de Cobertura: Minimizando riesgos en portafolios de alto rendimiento."</li>
                            <li>"Invitación a Foro de Inversión Élite: Discusión sobre capital de riesgo y mercados internacionales."</li>
                        </ul>
                    </div>
                """,
                "Estrategia 2": """
                    <div style="text-align: justify; font-size: 16px; color: black;width: 650px;border-right: 3px solid #b7b7b7 ; padding-right: 50px;">
                        <h3><span style="color: #38761d;font-weight: bold;">Comunicación para Clientes de Alto Riesgo con Estrategia 2</span></h3>
                        <p><span style="color: #1155cc; font-weight: bold;">Enfoque:</span> Estrategias de inversión complejas, planificación patrimonial y optimización fiscal.</p>
                        <p><span style="color: #1155cc; font-weight: bold;">Acciones Concretas:</span></p>
                        <ul>
                            <li>Asesor Patrimonial Dedicado: Un experto financiero asignado con disponibilidad para consultas frecuentes y seguimiento proactivo. Este asesor será su punto de contacto principal para todas las decisiones de inversión.</li>
                            <li>Sesiones de Planificación Patrimonial y Sucesoria: Consultas con especialistas para estructurar legados, optimizar impuestos y asegurar la transferencia eficiente de activos.</li>
                            <li>Análisis de Oportunidades de Inversión Alternativas: Presentación de fondos de capital privado, inversiones en startups, o proyectos de infraestructura, con análisis de due diligence y proyecciones detalladas.</li>
                        </ul>
                    </div>
                """,
                "Estrategia 3": """
                    <div style="text-align: justify; font-size: 16px; color: black;width: 650px;border-right: 3px solid #b7b7b7 ; padding-right: 50px;">
                        <h3><span style="color: #38761d;font-weight: bold;">Comunicación para Clientes de Alto Riesgo con Estrategia 3</span></h3>
                        <p><span style="color: #1155cc; font-weight: bold;">Ventajas Concretas:</span></p>
                        <ul>
                            <li>Acceso Prioritario a Oportunidades de Inversión: Ser los primeros en recibir información y tener la opción de participar en emisiones privadas de deuda, proyectos de inversión exclusivos o fondos de capital de riesgo.</li>
                            <li>Tarifas Preferenciales en Servicios de Gestión de Patrimonio: Descuentos del 20-25% en la gestión integral de sus portafolios de inversión, planeación fiscal avanzada y servicios de family office.</li>
                            <li>Invitaciones a Conferencias Internacionales de Inversión: Patrocinio o acceso preferencial a eventos de alto nivel donde puedan interactuar con grandes inversionistas y expertos financieros globales.</li>
                        </ul>
                    </div>
                """,
            }
        }

        with col_texto:
            selected_strategy = selected_images_text[0] if selected_images_text else None
            if opcion_col1 and selected_strategy:
                if opcion_col1 in communication_matrix and selected_strategy in communication_matrix[opcion_col1]:
                    specific_communication = communication_matrix[opcion_col1][selected_strategy]
                    st.markdown(specific_communication, unsafe_allow_html=True)
                else:
                    st.warning("No hay una comunicación específica definida para esta combinación. Se muestra un texto genérico.")
                    st.write(f"### Detalles para {opcion_col1} y la {', '.join(selected_images_text)}")
                    texto_explicativo = f"""
                    Aquí te presentamos información relevante basada en tus selecciones.

                    La opción **{opcion_col1}** suele estar asociada con el perfil de riesgo {opcion_col1.lower()}.

                    Mientras que la selección de {', '.join(selected_images_text)} indica una preferencia por aspectos relacionados con estas estrategias.

                    Esta combinación busca... (aquí puedes añadir más texto explicativo dinámico o un mensaje de fallback).
                    """
                    st.markdown(texto_explicativo)
            else:
                st.info("Selecciona un grupo de clientes y una estrategia para ver la comunicación específica.")

        #  {opcion_col1} y la {', '.join(selected_images_text)}")

        

        with col_grafico:
            st.markdown("### <font color='#38761d' style='font-weight: bold;'>Impacto Proyectado con Horizonte a 1 año</font>", unsafe_allow_html=True)

            current_group_data = base_imp[base_imp['Grupo'] == opcion_col1].iloc[0]
            selected_strategy = selected_images_text[0] if selected_images_text else None

            data = []

            # Determinar las columnas según la selección de estrategia
            if selected_strategy == "Estrategia 1":
                nombre_saldo = "saldo_con_estrategia1"
                nombre_des = "tasa_des1"
            elif selected_strategy == "Estrategia 2":
                nombre_saldo = "saldo_con_estrategia2"
                nombre_des = "tasa_des2"
            else:
                nombre_saldo = f"saldo_con_estrategia3"
                nombre_des = "tasa_des3"
            
            saldo_con_estrategia = current_group_data[nombre_saldo]
            saldo_sin_estrategia = current_group_data['saldo_sin_estrategia']
            #comision_con_estrategia = current_group_data["Comision_Sin_Estrategia"]
            tasa_desercion_con_estrategia = current_group_data[nombre_des]
            tasa_desercion_sin_estrategia = current_group_data['tasa_des']

            data = []

            # Bar for Saldo Sin Estrategia
            data.append(go.Bar(
                x=[saldo_sin_estrategia],
                y=["Saldo sin Estrategia"],
                orientation='h',
                marker_color='rgb(0,151,167)',
                name='Sin Estrategia',
                text=[f''],
                textposition='inside',
                #customdata=[{"tasa_desercion": tasa_desercion_sin_estrategia, "saldo": saldo_sin_estrategia }],
                #hovertemplate="<b>Con Estrategia</b><br>Valor: $%{customdata[0][saldo]:,.0f} MXN<br>Tasa de Deserción: %{customdata[0][tasa_desercion]:.2%}<extra></extra>"
            ))

            # Bar for Saldo Con Estrategia
            saldo_increase_percent = ((saldo_con_estrategia - saldo_sin_estrategia) / saldo_sin_estrategia) * 100
            saldo_text = f''
            if saldo_increase_percent > 0:
                saldo_text += f' (+{saldo_increase_percent:.1f}%)'

            data.append(go.Bar(
                x=[saldo_con_estrategia * 1.3],
                y=["Saldo con Estrategia"],
                orientation='h',
                marker_color='rgb(106,168,79)',
                name='Con Estrategia',
                text=[saldo_text],
                textposition='inside',
                #customdata=[{"tasa_desercion": tasa_desercion_con_estrategia, "saldo": saldo_con_estrategia }],
                #hovertemplate="<b>Con Estrategia</b><br>Valor: $%{customdata[0][saldo]:,.0f} MXN<br>Tasa de Deserción: %{customdata[0][tasa_desercion]:.2%}<extra></extra>"
                ))
            
            fig = go.Figure(data=data)

            fig.update_layout(
                title=f'',
                xaxis_title="Valor Proyectado (MDP)",
                yaxis_title="",
                yaxis={'categoryorder':'array', 'categoryarray': ['Comisión (Con Estrategia)', 'Comisión (Sin Estrategia)', 'Saldo Adm. (Con Estrategia)', 'Saldo Adm. (Sin Estrategia)']},
                height=300, # Altura de la figura
                margin=dict(l=10, r=100, t=50, b=50), # Adjust margins for better text display
                title_x=0.5, # Center the title
                legend= None
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**Observaciones:**")
            st.write(f"- Tasa de comisión: **7.5%** de comisión")
            st.write(f"- Tasa de rendimiento anual: **10.0%** **")
            st.write(f"- Información a mayo de 2025")

        current_group_data = base_imp[base_imp['Grupo'] == opcion_col1].iloc[0]
        selected_strategy = selected_images_text[0] if selected_images_text else None

        data = []

        # Determinar las columnas según la selección de estrategia
        if selected_strategy == "Estrategia 1":
            nombre_saldo = "saldo_con_estrategia1"
            nombre_des = "tasa_des1"
        elif selected_strategy == "Estrategia 2":
            nombre_saldo = "saldo_con_estrategia2"
            nombre_des = "tasa_des2"
        else:
            nombre_saldo = f"saldo_con_estrategia3"
            nombre_des = "tasa_des3"
        
        saldo_con_estrategia = current_group_data[nombre_saldo]
        saldo_sin_estrategia = current_group_data['saldo_sin_estrategia']
        #comision_con_estrategia = current_group_data["Comision_Sin_Estrategia"]
        tasa_desercion_con_estrategia = current_group_data[nombre_des]
        tasa_desercion_sin_estrategia = current_group_data['tasa_des']
 
        # Display the additional financial impact below the chart
            #---- agregamos la leyenda del saldo
        st.markdown(f"""
                <div style="display: flex; align-items: center; margin-top: 40px; margin-bottom: 20px;">
                    <hr style="flex-grow: 1.5; border-top: 2px solid #0492b8; margin-right: 10px;">
                    <div style="text-align: center;">
                        <span class="legend-saldo" style="font-size: 35px; font-weight: bold;">
                        Resultados Importantes</span>
                    </div>
                    <hr style="flex-grow: 1.5; border-top: 2px solid #0492b8; margin-left: 10px;">
                </div>
            """, unsafe_allow_html=True)

        st.markdown(f"**Ganancia Adicional por la {selected_strategy} en el grupo de {opcion_col1}:**")
        st.write(f"- Saldo Administrado Adicional Proyectado: **${(saldo_con_estrategia - saldo_sin_estrategia):,.0f} MDP**")
        st.write(f"- Comisión Adicional Generada Proyectada: **${((saldo_con_estrategia - saldo_sin_estrategia)*1000000*0.075):,.0f} MXN** (considerando una tasa de comisión del 8%).")


if __name__ == "__main__":
    run()
# aplicativo_streamlit.py
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os

# ----------------------------------------------------
# 1. Configurações e Carregamento de Dados/Modelo
# ----------------------------------------------------
st.set_page_config(page_title="People Analytics", layout="wide", initial_sidebar_state="expanded")

# Carregar modelo (Ajuste para rodar no Streamlit Cloud)
@st.cache_resource
def load_model():
    try:
        # Pega o caminho do diretório do arquivo atual
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Constrói o caminho completo para o arquivo do modelo
        model_path = os.path.join(base_dir, "modelo_rf.pkl")
        
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error("Erro: O arquivo 'modelo_rf.pkl' não foi encontrado. Certifique-se de que o modelo está no mesmo diretório do aplicativo.")
        st.stop()


# Carregar base de dados (para as listas de opções)
@st.cache_data
def load_data():
    try:
        # Pega o caminho do diretório do arquivo atual
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Constrói o caminho completo para o arquivo da base
        csv_path = os.path.join(base_dir, "rh_data.csv")
        return pd.read_csv(csv_path)
        
    except FileNotFoundError:
        st.error("Erro: O arquivo 'rh_data.csv' não foi encontrado. Certifique-se de que a base de dados está no mesmo diretório.")
        st.stop()

rf_model = load_model()
df = load_data()

# ----------------------------------------------------
# 2. Layout da Barra Lateral (Sidebar)
# ----------------------------------------------------
with st.sidebar:
    st.title("💡 Sobre este App")
    st.markdown("""
    Este aplicativo utiliza **Machine Learning** para prever a probabilidade de um colaborador solicitar desligamento. O modelo **Random Forest** foi treinado com dados históricos para gerar previsões precisas e confiáveis.
    
    Ele fornece **insights práticos e ações de retenção personalizadas**, auxiliando líderes e equipes de RH na tomada de decisão estratégica e no fortalecimento do engajamento.
    """)
    st.markdown("---")
    st.header("📚 Entenda a Tecnologia")
    st.markdown("""
    - **Machine Learning**: Permite que computadores identifiquem padrões a partir de dados, sem programação explícita.
    - **Random Forest**: Algoritmo que combina múltiplas árvores de decisão para criar um modelo mais robusto e preciso.
    

    **Por que a retenção é importante?**

    Reter talentos é crucial para o crescimento e a sustentabilidade. A alta rotatividade gera custos significativos, impacta a moral da equipe e causa perda de conhecimento. Investir em retenção é entender e valorizar o capital humano da empresa.
    """)
    st.markdown("---")
    st.info("Desenvolvido por Vanessa Santana do Amaral\n\n"

            "Github: https://github.com/vanessasantanadoamaral\n\n"

            "Linkedin: https://www.linkedin.com/in/vanessasantanadoamaral/)")

# ----------------------------------------------------
# 3. Seção Principal: Inputs do Usuário
# ----------------------------------------------------
st.title("People Analytics - Previsão de Churn")
st.subheader("👤 Análise de Risco Individual")
st.markdown("Preencha os dados do colaborador para prever a probabilidade de desligamento.")

with st.form("input_funcionario"):
    st.markdown("---")
    
    # Organização dos inputs em colunas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("Informações Pessoais")
        Age = st.number_input("Idade", min_value=18, max_value=70, value=30)
        
        # Ajuste: Tradução e ordenação dos tipos de viagem
        travel_map = {"Travel_Frequently": "Viaja Frequentemente", "Travel_Rarely": "Viaja Raramente", "Non-Travel": "Não Viaja"}
        BusinessTravel = st.selectbox(
            "Viagens a trabalho", 
            options=sorted(list(travel_map.keys())),
            format_func=lambda x: travel_map[x]
        )
        
        DistanceFromHome = st.number_input("Distância de casa (km)", min_value=0, max_value=100)
        
        
        Education = st.selectbox(
            "Nível de Educação", 
            df["Education"].unique(),
            help="1: Abaixo da faculdade, 2: Universidade, 3: Bacharelado, 4: Mestrado, 5: Doutorado"
        )
    
    with col2:
        st.header("Informações Profissionais")
        
        # Ajuste: Tradução e ordenação dos departamentos
        department_map = {"Human Resources": "Recursos Humanos", "Research & Development": "Pesquisa & Desenvolvimento", "Sales": "Vendas"}
        Department = st.selectbox(
            "Departamento", 
            options=sorted(list(department_map.keys())),
            format_func=lambda x: department_map[x]
        )
        
        # Nível do cargo e expander para a descrição
        JobLevel = st.selectbox("Nível do cargo", df["JobLevel"].unique(),
        help="1: Junior, 2: Pleno, 3: Sênior, 4: Coordenador, 5: Gerente / Diretor")
        
        # Ajuste: Tradução e ordenação dos cargos
        jobrole_map = {
            "Healthcare Representative": "Representante de Saúde",
            "Human Resources": "Recursos Humanos",
            "Laboratory Technician": "Técnico de Laboratório",
            "Manager": "Gerente",
            "Manufacturing Director": "Diretor de Manufatura",
            "Research Director": "Diretor de Pesquisa",
            "Research Scientist": "Cientista de Pesquisa",
            "Sales Executive": "Executivo de Vendas",
            "Sales Representative": "Representante de Vendas"
        }
        JobRole = st.selectbox(
            "Cargo",
            options=sorted(list(jobrole_map.keys())),
            format_func=lambda x: jobrole_map[x]
        )
        
        MonthlyIncome = st.number_input("Salário Mensal", min_value=1000, value=5000)
        PercentSalaryHike = st.number_input("Percentual de aumento salarial (%)", min_value=0)
    
    with col3:
        st.header("Histórico na Empresa")
        YearsAtCompany = st.number_input("Tempo na empresa (anos)", min_value=0)
        YearsSinceLastPromotion = st.number_input("Anos desde a última promoção", min_value=0)
        YearsWithCurrManager = st.number_input("Anos com o gestor atual", min_value=0)
        TotalWorkingYears = st.number_input("Total de anos de experiência", min_value=0)
        NumCompaniesWorked = st.number_input("Número de empresas anteriores", min_value=0)
        TrainingTimesLastYear = st.number_input("Treinamentos no último ano", min_value=0)
        
        # Ajuste: Descrição do nível de ações
        StockOptionLevel = st.number_input(
            "Nível de opções de ações", 
            min_value=0, 
            help="Quanto maior o número, mais opções de ações o funcionário possui."
        )
        
    st.markdown("---")
    submitted = st.form_submit_button("📈 Prever Desligamento", use_container_width=True)

# ----------------------------------------------------
# 4. Lógica de Previsão e Exibição de Resultados
# ----------------------------------------------------
if submitted:
    with st.spinner("Analisando os dados..."):
        # Mapeamento para garantir que os inputs correspondem aos dados de treinamento do modelo
        input_dict = {
            "Age": [Age], "BusinessTravel": [BusinessTravel], "Department": [Department],
            "DistanceFromHome": [DistanceFromHome], "Education": [Education], "JobLevel": [JobLevel],
            "JobRole": [JobRole], "MonthlyIncome": [MonthlyIncome], "NumCompaniesWorked": [NumCompaniesWorked],
            "PercentSalaryHike": [PercentSalaryHike], "StockOptionLevel": [StockOptionLevel],
            "TotalWorkingYears": [TotalWorkingYears], "TrainingTimesLastYear": [TrainingTimesLastYear],
            "YearsAtCompany": [YearsAtCompany], "YearsSinceLastPromotion": [YearsSinceLastPromotion],
            "YearsWithCurrManager": [YearsWithCurrManager]
        }
        input_df = pd.DataFrame(input_dict)

        # One-hot encoding e alinhamento de colunas
        cat_cols = ["BusinessTravel", "Department", "JobRole"]
        input_df_encoded = pd.get_dummies(input_df, columns=cat_cols)

        model_features = rf_model.feature_names_in_ if hasattr(rf_model, 'feature_names_in_') else None
        if model_features is None:
            st.error("Erro no modelo: 'feature_names_in_' não encontrado.")
            st.stop()
            
        missing_cols = set(model_features) - set(input_df_encoded.columns)
        for c in missing_cols:
            input_df_encoded[c] = 0
        input_df_encoded = input_df_encoded[model_features]

        # Previsão
        prob = rf_model.predict_proba(input_df_encoded)[0][1]
        prob_percent = round(prob * 100, 2)

    st.markdown("---")
    st.subheader("📊 Resultado da Análise de Churn")
    
    # Definição dos níveis de risco
    RISCO_BAIXO_MAX = 30
    RISCO_MEDIO_MAX = 60

    if prob_percent <= RISCO_BAIXO_MAX:
        status_text = "✅ **BAIXO RISCO** de desligamento."
        status_color = "#4CAF50" # Verde
        recom_title = "Ações para manter o engajamento e a satisfação"
        recom_list = """
        - **Monitoramento contínuo:** Realize check-ins periódicos para garantir que o colaborador continue satisfeito e engajado.
        - **Oportunidades de crescimento:** Continue oferecendo desafios e oportunidades de aprendizado para manter o alto desempenho.
        - **Reconhecimento:** Reconheça e celebre as conquistas para reforçar o sentimento de valorização.
        - **Qualidade de vida:** Estimule o equilíbrio entre vida pessoal e profissional.
        """
    elif prob_percent <= RISCO_MEDIO_MAX:
        status_text = "⚠️ **MÉDIO RISCO** de desligamento. Atenção!"
        status_color = "#FFC107" # Laranja
        recom_title = "Ações sugeridas para mitigar o risco"
        recom_list = """
        - **Diálogo aberto:** Agende uma conversa com o colaborador para entender suas expectativas e possíveis insatisfações.
        - **Plano de desenvolvimento:** Crie um plano de carreira individualizado (PDI), mostrando um caminho claro de crescimento na empresa.
        - **Análise de remuneração:** Revise a competitividade do salário e dos benefícios.
        - **Mentoria:** Conecte o colaborador com um mentor para oferecer orientação e apoio.
        - **Feedback construtivo:** Promova feedbacks mais frequentes e orientados para o desenvolvimento.
        """
    else:
        status_text = "🚨 **ALTO RISCO** de desligamento. Aja rapidamente!"
        status_color = "#E53935" # Vermelho
        recom_title = "Ações de retenção críticas"
        recom_list = """
        - **Desenvolvimento e Carreira:** Crie um plano de desenvolvimento individual com metas claras e urgentes. Ofereça mentoria ou coaching.
        - **Reconhecimento e Valorização:** Destaque suas conquistas e revise urgentemente o plano de remuneração e benefícios.
        - **Engajamento e Clima:** Realize um check-in profundo para ouvir as necessidades e, se possível, flexibilize a jornada de trabalho.
        - **Liderança e Gestão:** Capacite o gestor para práticas de liderança mais empáticas e garanta que a carga de trabalho seja justa.
        """

    # Exibição do gráfico e métricas
    col_chart, col_status = st.columns([0.6, 0.4])

    with col_chart:
        fig = go.Figure(data=[go.Pie(
            labels=["Probabilidade de Churn", "Probabilidade de Permanência"],
            values=[prob_percent, 100 - prob_percent],
            hole=0.7,
            marker=dict(colors=[status_color, "#d9d9d9"]),
            textinfo="label+percent"
        )])
        fig.update_layout(
            title_text="Probabilidade de Desligamento",
            title_x=0.5,
            showlegend=False,
            height=300,
            margin=dict(t=50, b=0, l=0, r=0),
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_status:
        st.subheader("Resultado da Análise")
        st.markdown(f"#### {status_text}")
        st.markdown(f"**Probabilidade de Churn:** `{prob_percent}%`")
        st.markdown("---")
        st.info("O modelo fornece uma estimativa. Use-a como um ponto de partida para um diálogo construtivo.")

    # Seção de recomendações dinâmicas
    with st.container(border=True):
        st.subheader(f"🚀 {recom_title}")
        st.markdown(recom_list)
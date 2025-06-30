import streamlit as st
import numpy as np
from scipy.integrate import dblquad, quad
from PIL import Image
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
import os
import google.generativeai as genai

# --- Configuração Central dos Agentes ---
AGENTS = {
    "Cris":     {"prefix": "publico", "icon": "🌍"},
    "Alex":     {"prefix": "restrito", "icon": "🔒"},
    "Guga":     {"prefix": "especialista", "icon": "🎓"}
}

def get_knowledge_filepaths(prefix):
    """Retorna os nomes dos arquivos para um determinado prefixo de IA."""
    return {
        "vectorizer": f"{prefix}_vectorizer.pkl",
        "matrix": f"{prefix}_matriz_tfidf.pkl",
        "parts": f"{prefix}_partes_texto.pkl"
    }

def guardar_conocimiento(prefix, vectorizer, matriz_tfidf, partes_texto):
    """Guarda os objetos processados para um IA específico."""
    filepaths = get_knowledge_filepaths(prefix)
    try:
        with open(filepaths["vectorizer"], "wb") as f: pickle.dump(vectorizer, f)
        with open(filepaths["matrix"], "wb") as f: pickle.dump(matriz_tfidf, f)
        with open(filepaths["parts"], "wb") as f: pickle.dump(partes_texto, f)
        return True
    except Exception as e:
        st.error(f"Erro ao salvar conhecimento para '{prefix}': {e}")
        return False

def cargar_conocimiento(prefix):
    """Carga os objetos processados para um IA específico se existirem."""
    filepaths = get_knowledge_filepaths(prefix)
    if not all(os.path.exists(fp) for fp in filepaths.values()):
        return None, None, None
    try:
        with open(filepaths["vectorizer"], "rb") as f: vectorizer = pickle.load(f)
        with open(filepaths["matrix"], "rb") as f: matriz_tfidf = pickle.load(f)
        with open(filepaths["parts"], "rb") as f: partes_texto = pickle.load(f)
        return vectorizer, matriz_tfidf, partes_texto
    except Exception as e:
        st.error(f"Erro ao carregar conhecimento para '{prefix}': {e}")
        return None, None, None

def extrair_texto_pdf(arquivo_pdf):
    if arquivo_pdf is None: return ""
    try:
        with pdfplumber.open(arquivo_pdf) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo PDF: {e}")
        return ""

def dividir_texto_em_partes(texto, max_tokens=200):
    texto = re.sub(r'\s+', ' ', texto).strip()
    sentencas = re.split(r'(?<=[.!?])\s+', texto)
    partes, parte_atual = [], ""
    for sentenca in sentencas:
        if len(parte_atual.split()) + len(sentenca.split()) <= max_tokens:
            parte_atual += sentenca + " "
        else:
            if parte_atual: partes.append(parte_atual.strip())
            parte_atual = sentenca + " "
    if parte_atual: partes.append(parte_atual.strip())
    return partes

def encontrar_melhor_resposta_agente(pergunta_usuario, partes_texto, vectorizer, matriz_tfidf):
    if not partes_texto or vectorizer is None or matriz_tfidf is None:
        return "Não tenho o conhecimento necessário para participar deste debate."
    vetor_pergunta = vectorizer.transform([pergunta_usuario])
    similaridades = cosine_similarity(vetor_pergunta, matriz_tfidf)
    indice_melhor_parte = np.argmax(similaridades)
    if similaridades[0, indice_melhor_parte] > 0.1:
        return partes_texto[indice_melhor_parte]
    else:
        return "Não encontrei informações relevantes sobre este tópico no meu documento."

def chamar_api_gemini(prompt_usuario, contexto_agentes):
   
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        
    except (KeyError, FileNotFoundError):
        st.error("Chave da API do Gemini não encontrada. Por favor, configure-a nos segredos do Streamlit.")
        return "A API do Gemini não está configurada corretamente."

    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt_completo = f"""
    **Instrução:** Você é um mediador de debate. Sua tarefa é analisar a pergunta de um usuário e as perspectivas de diferentes agentes especialistas. Com base nessas informações, formule uma resposta abrangente e bem fundamentada em português.

    **Pergunta do Usuário:**
    {prompt_usuario}

    **Perspectivas dos Agentes (Contexto):**
    {contexto_agentes}

    **Sua Resposta (como mediador):**
    """
    try:
        resposta = model.generate_content(prompt_completo)
        return resposta.text
    except Exception as e:
        st.error(f"Erro ao chamar a API do Gemini: {e}")
        return "Ocorreu um erro ao gerar a resposta do debate."

def iniciar_debate(prompt, knowledge_base):
    """
    Inicia o debate, coleta as perspectivas dos agentes e usa a API do Gemini
    para gerar uma resposta final.
    """
    contexto_para_gemini = ""
    participantes = 0
    for name, info in AGENTS.items():
        k = knowledge_base[info["prefix"]]
        if all(v is not None for v in k.values()):
            participantes += 1
            resposta_agente = encontrar_melhor_resposta_agente(prompt, k["p"], k["v"], k["m"])
            contexto_para_gemini += f"---\n### {info['icon']} **{name}** diz:\n> {resposta_agente.replace('\n', '\n> ')}\n\n"

    if participantes == 0:
        return "Nenhum assistente com conhecimento carregado. Por favor, processe os PDFs na barra lateral para que possamos discutir."

    resposta_final = chamar_api_gemini(prompt, contexto_para_gemini)
    return resposta_final


def setup_sidebar_knowledge_manager():
    with st.sidebar:
        st.header("📚 Gerenciador de Conhecimento do Chatbot")
        st.write("Carregue os PDFs para cada agente especialista aqui.")
        for name, info in AGENTS.items():
            with st.expander(f"{info['icon']} Especialista: {name}", expanded=(name == "Cris")):
                pdf_file = st.file_uploader(f"Carregar PDF para {name}", key=f"pdf_{info['prefix']}", type="pdf")
                if st.button(f"Processar PDF de {name}", key=f"btn_{info['prefix']}"):
                    if pdf_file:
                        with st.spinner(f"Processando PDF para {name}..."):
                            texto = extrair_texto_pdf(pdf_file)
                            if texto:
                                partes = dividir_texto_em_partes(texto)
                                if partes:
                                    vectorizer = TfidfVectorizer()
                                    matriz = vectorizer.fit_transform(partes)
                                    if guardar_conocimiento(info['prefix'], vectorizer, matriz, partes):
                                        st.session_state.knowledge[info['prefix']] = {"v": vectorizer, "m": matriz, "p": partes}
                                        st.success(f"Conhecimento para {name} salvo e ativado!")
                                        st.rerun()
                                else:
                                    st.error(f"Não foi possível dividir o texto para {name}.")
                            else:
                                st.error(f"Não foi possível extrair texto do PDF para {name}.")
                    else:
                        st.warning(f"Carregue um arquivo PDF para {name} primeiro.")


def run_web_application():
    st.header("APLICAÇÃO WEB: Modelo de Custo-Taxa de Manutenção")
    st.subheader("Insira os seguintes parâmetros")
    col1, col2 = st.columns(2)
    with col1:
        n1 = st.number_input(f"Parâmetros de escala dos componentes fracos - η1", min_value=0.0, value=0.3)
        b1 = st.number_input(f"Parâmetro de forma dos componentes fracos - β1", min_value=1.0, max_value=6.0, value=3.0)
        a = st.number_input(f"Parâmetro de mistura - α", min_value=0.0, max_value=1.0, value=0.05)
        u = st.number_input(f"Taxa de chegada de choques - μ", min_value=0.0, max_value=2.0, value=0.5)
        cf = st.number_input("Custo de manutenção corretiva - CF", min_value=0.0, value=5.0)
        cn = st.number_input("Custo unitário de degradação natural - CN", min_value=0.0, value=0.04)
    with col2:
        n2 = st.number_input(f"Parâmetros de escala dos componentes fortes - η2", min_value=0.0, value=3.0)
        b2 = st.number_input(f"Parâmetro de forma dos componentes fortes - β2", min_value=1.0, max_value=6.0, value=3.0)
        l = st.number_input(f"Recíproco do tempo médio de atraso - λ", min_value=0.1, value=1.0)
        ci = st.number_input("Custo de inspeção - CI", min_value=0.0, value=0.1)
        cp = st.number_input("Custo de manutenção preventiva - CP", min_value=0.0, value=1.0)
        cc = st.number_input("Custo unitário de degradação por choque - CS", min_value=0.0, value=0.04)
    st.subheader("Insira as seguintes variáveis de decisão")
    k = int(st.number_input("Número de inspeções - K", min_value=0, max_value=30, step=1, value=8))
    d = st.number_input(f"Frequência de inspeção - Δ", min_value=0.000, value=0.370, format="%.3f")
    t = st.number_input("Idade para manutenção preventiva - T", min_value=0.000, value=3.25, format="%.3f")
    st.subheader("Clique no botão abaixo para executar esta aplicação:")
    if st.button("Obter Custo-Taxa"):
        with st.spinner('Calculando... Por favor, aguarde.'):
            y = [k, d, t]
            cost_rate = calculate_cost_rate(y, n1, n2, b1, b2, a, l, u, cf, ci, cn, cc, cp)
            st.success(f"Cálculo finalizado!")
            st.metric(label="Custo-taxa", value=f"{cost_rate:.4f}")

def run_chatbot_app():
    st.header("🤖 Chatbot com Discussão de Agentes Especialistas", divider=True)
    st.write("Por favor, faça uma pergunta e os assistentes debaterão uma resposta com base em seus conhecimentos.")

    # Exibe histórico de chat
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []
        
    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"], avatar=msg.get("avatar", "🤖")):
            st.write(msg["content"])

    # Input do usuário
    if prompt := st.chat_input("Faça sua pergunta para o painel de especialistas..."):
        st.session_state.mensagens.append({"role": "user", "content": prompt, "avatar": "👤"})
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Os assistentes estão debatendo..."):
                resposta_debate = iniciar_debate(prompt, st.session_state.knowledge)
                st.write(resposta_debate)

        st.session_state.mensagens.append({"role": "assistant", "content": resposta_debate, "avatar": "🤖"})


def main():
    # --- Configuração da página ---
    st.set_page_config(
        page_title="App de Manutenção & Chatbot",
        page_icon="🔧",
        layout="centered"
    )

    # Inicialização do estado de sessão para o chatbot
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []
    if "knowledge" not in st.session_state:
        st.session_state.knowledge = {
            agent_info["prefix"]: {"v": None, "m": None, "p": None}
            for agent_info in AGENTS.values()
        }
    if "init_load_done" not in st.session_state:
        st.session_state.init_load_done = True
        for name, info in AGENTS.items():
            v, m, p = cargar_conocimiento(info["prefix"])
            if v is not None and m is not None and p is not None:
                st.session_state.knowledge[info['prefix']] = {"v": v, "m": m, "p": p}
                st.toast(f"Conhecimento '{name}' carregado do disco.", icon=info["icon"])

    # --- Configuração da Barra Lateral ---
    setup_sidebar_knowledge_manager()

    # --- Título e Imagem Principal ---
    try:
        foto = Image.open('foto.png')
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(foto, use_container_width=True)
    except FileNotFoundError:
        st.warning("Aviso: 'foto.png' não encontrada.")

    st.title('SOFTWARE DE POLÍTICA DE MANUTENÇÃO')

    st.divider()
    with st.expander("Ver Informações do Modelo"):
        st.write('''
        A política de manutenção para a motobomba, sujeita à degradação natural e a choques que a colocam em estado defeituoso, é aprimorada. Um custo adicional proporcional ao tempo que o sistema permanece em condição defeituosa é introduzido, com custos diferenciados dependendo da origem do defeito. Resultados computacionais são obtidos através da otimização da frequência de inspeções, do intervalo entre inspeções consecutivas e do período recomendado para a realização da manutenção preventiva.
        ''')
        st.subheader("Modelo de Taxa de Custo")
        st.latex(r"C_{\infty} = (K, \Delta, T) = \frac{\sum_{s=1}^{9} E_{Cs}}{\sum_{s=1}^{9} E_{Ls}}")
        st.markdown(r"Onde $E[C]$ é o custo esperado e $E[L]$ é a duração esperada do ciclo.")
        st.subheader("Funções de Distribuição")
        st.markdown(r"""
        **1. Degradação (Weibull):**

        $$
        f_X(x) = \alpha \left( \frac{\beta_1}{\eta_1} \left( \frac{x}{\eta_1} \right)^{\beta_1 - 1} e^{- \left( \frac{x}{\eta_1} \right)^{\beta_1}} \right)
        + (1 - \alpha) \left( \frac{\beta_2}{\eta_2} \left( \frac{x}{\eta_2} \right)^{\beta_2 - 1} e^{- \left( \frac{x}{\eta_2} \right)^{\beta_2}} \right)
        $$
        """)

        st.markdown(r"**2. Choques (Exponencial):** $f_Z(z) = \mu e^{-\mu z}$")
        st.markdown(r"**3. Tempo de Atraso (Exponencial):** $f_H(h) = \lambda e^{-\lambda h}$")

    # --- SEÇÃO 1: MODELO MATEMÁTICO ---
    run_web_application()

    # --- Separador Visual ---
    st.divider()

    # --- SEÇÃO 2: CHATBOT ---
    run_chatbot_app()

    # --- SEÇÃO 3: INFORMAÇÃO ADICIONAL (em expanders) ---

    with st.expander("Ver Informações do Site"):
        st.write("O RANDOM - Grupo de Pesquisa em Risco e Análise de Decisões em Operações e Manutenção, foi criado em 2012 com o objetivo de reunir diferentes pesquisadores que atuam nas áreas de risco e modelagem em manutenção e operações.")
        st.markdown('[Clique aqui para ser redirecionado para o site oficial](https://sites.ufpe.br/random/)', unsafe_allow_html=True)


# Função de cálculo do modelo (sem alterações)
def calculate_cost_rate(y, n1, n2, b1, b2, a, l, u, cf, ci, cn, cc, cp):
    K, D, T = y[0], y[1], y[2]
    f01 = lambda x: (b1 / n1) * ((x / n1)**(b1 - 1)) * np.exp(-(x / n1)**b1)
    f02 = lambda x: (b2 / n2) * ((x / n2)**(b2 - 1)) * np.exp(-(x / n2)**b2)
    fx = lambda x: (a * f01(x)) + ((1 - a) * f02(x))
    Rx = lambda t_val: quad(fx, t_val, np.inf)[0]
    fz = lambda z: u * np.exp(-u * z)
    Rz = lambda z: np.exp(-u * z)
    fh = lambda h: l * np.exp(-l * h)
    Rh = lambda h: np.exp(-l * h)
    sc = sv = 0.0
    for i in range(1, K + 1):
        lim_inf, lim_sup = (i - 1) * D, i * D
        c1, _ = dblquad(lambda h, x: (cf + ci * (i - 1) + cn * h) * Rz(x) * fx(x) * fh(h), lim_inf, lim_sup, 0, lambda x: lim_sup - x)
        t1, _ = dblquad(lambda x, h: (x + h) * Rz(x) * fx(x) * fh(h), lim_inf, lim_sup, 0, lambda x: lim_sup - x)
        sc += c1
        sv += t1
        c2, _ = quad(lambda x: (cp + ci * i + cn * (lim_sup - x)) * Rz(x) * fx(x) * Rh(lim_sup - x), lim_inf, lim_sup)
        t2, _ = quad(lambda x: lim_sup * Rz(x) * fx(x) * Rh(lim_sup - x), lim_inf, lim_sup)
        sc += c2
        sv += t2
    if T > K * D:
        c3, _ = dblquad(lambda h, x: (cf + ci * K + cn * h) * Rz(x) * fx(x) * fh(h), K * D, T, 0, lambda x: T - x)
        t3, _ = dblquad(lambda x, h: (x + h) * Rz(x) * fx(x) * fh(h), K * D, T, 0, lambda x: T - x)
        sc += c3
        sv += t3
        c4, _ = quad(lambda x: (cp + ci * K + cn * (T - x)) * Rz(x) * fx(x) * Rh(T - x), K * D, T)
        t4, _ = quad(lambda x: T * Rz(x) * fx(x) * Rh(T - x), K * D, T)
        sc += c4
        sv += t4
    for i in range(1, K + 1):
        lim_inf, lim_sup = (i - 1) * D, i * D
        c5, _ = dblquad(lambda h, z: (cf + ci * (i - 1) + cc * h) * Rx(z) * fz(z) * fh(h), lim_inf, lim_sup, 0, lambda z: lim_sup - z)
        t5, _ = dblquad(lambda z, h: (z + h) * Rx(z) * fz(z) * fh(h), lim_inf, lim_sup, 0, lambda z: lim_sup - z)
        sc += c5
        sv += t5
        c6, _ = quad(lambda z: (cp + ci * i + cc * (lim_sup - z)) * Rx(z) * fz(z) * Rh(lim_sup - z), lim_inf, lim_sup)
        t6, _ = quad(lambda z: lim_sup * Rx(z) * fz(z) * Rh(lim_sup - z), lim_inf, lim_sup)
        sc += c6
        sv += t6
    if T > K * D:
        c7, _ = dblquad(lambda h, z: (cf + ci * K + cc * h) * Rx(z) * fz(z) * fh(h), K * D, T, 0, lambda z: T - z)
        t7, _ = dblquad(lambda z, h: (z + h) * Rx(z) * fz(z) * fh(h), K * D, T, 0, lambda z: T - z)
        sc += c7
        sv += t7
        c8, _ = quad(lambda z: (cp + ci * K + cc * (T - z)) * Rx(z) * fz(z) * Rh(T - z), K * D, T)
        t8, _ = quad(lambda z: T * Rx(z) * fz(z) * Rh(T - z), K * D, T)
        sc += c8
        sv += t8
    p9 = Rz(T) * Rx(T)
    c9 = (cp + ci * K) * p9
    t9 = T * p9
    sc += c9
    sv += t9
    return sc / sv if sv != 0 else 0

# --- Ponto de entrada do script ---
if __name__ == '__main__':
    main()
import os
import time
import numpy as np
import pygad
import ezdxf
from ezdxf import path
from shapely.geometry import Polygon, LineString, box
from shapely.ops import polygonize, unary_union
from shapely.affinity import rotate, translate
from shapely.prepared import prep
import matplotlib.pyplot as plt
import threading
import queue
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==============================================================================
# PARÂMETROS DA CHAPA E DA CNC
# ==============================================================================
LARGURA_CHAPA = 1000.0       # mm
ALTURA_CHAPA = 500.0         # mm
DIAMETRO_FRESA = 3.0         # mm
MARGEM_SEGURANCA = 1.0       # mm
OFFSET_TOTAL = (DIAMETRO_FRESA / 2.0) + (MARGEM_SEGURANCA / 2.0)

ANGULOS_PERMITIDOS = [0, 45, 90, 135, 180, 225, 270, 315]
ARQUIVO_SAIDA_DXF = "chapa_otimizada.dxf"
TOLERANCIA_FECHAMENTO = 0.01  # mm: precisão usada para unir extremidades próximas


# ==============================================================================
# 1. LEITURA E EXPORTAÇÃO DE DXF
# ==============================================================================
def carregar_poligono_de_dxf(
    caminho_arquivo,
    precisao_curva=0.5,
    tolerancia_fechamento=TOLERANCIA_FECHAMENTO
):
    """Lê entidades de um DXF e reconstrói o maior polígono fechado.

    As coordenadas são arredondadas para a tolerância informada antes da união
    das linhas. Isso corrige pequenas diferenças numéricas entre extremidades
    que visualmente parecem conectadas no CAD.
    """
    if tolerancia_fechamento <= 0:
        raise ValueError("A tolerância de fechamento deve ser positiva.")

    doc = ezdxf.readfile(caminho_arquivo)
    msp = doc.modelspace()
    linhas = []

    for entidade in msp:
        try:
            caminho = path.make_path(entidade)
            pontos = list(caminho.flattening(distance=precisao_curva))
            if len(pontos) < 2:
                continue

            coords = []
            for ponto in pontos:
                x = round(ponto.x / tolerancia_fechamento) * tolerancia_fechamento
                y = round(ponto.y / tolerancia_fechamento) * tolerancia_fechamento
                coordenada = (x, y)
                if not coords or coordenada != coords[-1]:
                    coords.append(coordenada)

            # Uma LineString precisa ter pelo menos dois pontos distintos.
            if len(set(coords)) >= 2:
                linha = LineString(coords)
                if linha.length > 0:
                    linhas.append(linha)

        except Exception:
            # Textos, cotas e entidades incompatíveis não fazem parte do contorno.
            continue

    if not linhas:
        raise ValueError(f"Nenhuma geometria válida encontrada em {caminho_arquivo}")

    uniao = unary_union(linhas)
    poligonos = list(polygonize(uniao))

    if not poligonos:
        raise ValueError(
            f"Contorno não está fechado em {caminho_arquivo}. "
            f"Verifique a geometria ou aumente cuidadosamente a tolerância."
        )

    poligono_principal = max(poligonos, key=lambda poligono: poligono.area)
    if poligono_principal.is_empty or poligono_principal.area <= 0:
        raise ValueError(f"O DXF não produziu um polígono com área válida: {caminho_arquivo}")

    minx, miny, _, _ = poligono_principal.bounds
    return translate(poligono_principal, xoff=-minx, yoff=-miny)


def exportar_layout_para_dxf(
    pecas_originais_posicionadas,
    largura_chapa,
    altura_chapa,
    arquivo_saida
):
    """Exporta a chapa e os contornos originais posicionados para um novo DXF."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    doc.layers.add(name="CHAPA", color=3)
    doc.layers.add(name="CORTE", color=1)
    doc.layers.add(name="FUROS", color=5)

    msp.add_lwpolyline(
        [(0, 0), (largura_chapa, 0), (largura_chapa, altura_chapa), (0, altura_chapa)],
        close=True,
        dxfattribs={"layer": "CHAPA"}
    )

    for poligono in pecas_originais_posicionadas:
        coordenadas_externas = list(poligono.exterior.coords)
        msp.add_lwpolyline(
            coordenadas_externas,
            close=True,
            dxfattribs={"layer": "CORTE"}
        )

        for interior in poligono.interiors:
            coordenadas_internas = list(interior.coords)
            msp.add_lwpolyline(
                coordenadas_internas,
                close=True,
                dxfattribs={"layer": "FUROS"}
            )

    try:
        doc.saveas(arquivo_saida)
        caminho_salvo = arquivo_saida
    except PermissionError:
        raiz, extensao = os.path.splitext(arquivo_saida)
        sufixo = time.strftime("%Y%m%d_%H%M%S")
        caminho_alternativo = f"{raiz}_{sufixo}{extensao}"
        doc.saveas(caminho_alternativo)
        caminho_salvo = caminho_alternativo
        print(
            "Aviso: não foi possível sobrescrever o DXF anterior. "
            f"Uma nova cópia foi salva em: {caminho_salvo}"
        )

    print(f"Arquivo CAM exportado com sucesso: {caminho_salvo}")
    return caminho_salvo


# ==============================================================================
# 2. DECODIFICADOR GEOMÉTRICO (BOTTOM-LEFT)
# ==============================================================================
chapa = box(0, 0, LARGURA_CHAPA, ALTURA_CHAPA)


def decodificar_layout(ordem, indices_angulos, pecas_offset, pecas_originais):
    """Posiciona cada cópia de peça usando sua geometria com offset."""
    posicionadas_offset = []
    posicionadas_originais = []
    passo_busca = 10.0  # mm

    for idx_peca, idx_ang in zip(ordem, indices_angulos):
        idx = int(idx_peca)
        angulo = ANGULOS_PERMITIDOS[int(idx_ang)]

        pivo = pecas_originais[idx].centroid
        p_off = rotate(pecas_offset[idx], angulo, origin=pivo)
        p_orig = rotate(pecas_originais[idx], angulo, origin=pivo)

        minx, miny, maxx, maxy = p_off.bounds
        p_off = translate(p_off, xoff=-minx, yoff=-miny)
        p_orig = translate(p_orig, xoff=-minx, yoff=-miny)

        largura_p = maxx - minx
        altura_p = maxy - miny
        obstaculos = [prep(poligono) for poligono in posicionadas_offset]
        encaixou = False

        for x in np.arange(0, LARGURA_CHAPA - largura_p + 1, passo_busca):
            for y in np.arange(0, ALTURA_CHAPA - altura_p + 1, passo_busca):
                candidato_offset = translate(p_off, xoff=x, yoff=y)

                if chapa.contains(candidato_offset):
                    if not any(obs.intersects(candidato_offset) for obs in obstaculos):
                        posicionadas_offset.append(candidato_offset)
                        posicionadas_originais.append(translate(p_orig, xoff=x, yoff=y))
                        encaixou = True
                        break
            if encaixou:
                break

        if not encaixou:
            return None, None

    return posicionadas_offset, posicionadas_originais


# ==============================================================================
# 3. INTERFACE GRÁFICA
# ==============================================================================
class AplicacaoNesting:
    def __init__(self, root):
        self.root = root
        self.root.title("Otimizador de Nesting")
        self.root.geometry("1320x920")
        self.root.minsize(1050, 780)

        self.fila = queue.Queue()
        self.thread_otimizacao = None
        self.status_var = tk.StringVar(value="Pronto para iniciar a otimização.")
        self.progresso_var = tk.StringVar(value="0 / 60 gerações")
        self.resultado_var = tk.StringVar(value="")
        self.pasta_origem_var = tk.StringVar(value="Nenhuma pasta selecionada.")
        self.contagem_pecas_var = tk.StringVar(
            value="Selecione uma pasta para carregar os arquivos DXF."
        )
        self.quantidade_var = tk.StringVar(value="1")

        self.pasta_origem = None
        self.arquivos_dxf = []
        self.quantidades_por_iid = {}
        self.iid_para_arquivo = {}

        self._montar_interface()
        self._atualizar_selecao()
        self.root.after(100, self._processar_fila)

    def _montar_interface(self):
        estilo = ttk.Style()
        try:
            estilo.theme_use("clam")
        except tk.TclError:
            pass

        topo = ttk.Frame(self.root, padding=(18, 14))
        topo.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(
            topo,
            text="OTIMIZAÇÃO DE NESTING",
            font=("Segoe UI", 17, "bold")
        ).pack(anchor="w")
        ttk.Label(
            topo,
            text="Distribuição de perfis DXF em uma chapa de material",
            font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(2, 10))

        quadro_entrada = ttk.LabelFrame(topo, text="Peças de entrada", padding=(10, 7))
        quadro_entrada.pack(fill=tk.X, pady=(0, 10))

        linha_pasta = ttk.Frame(quadro_entrada)
        linha_pasta.pack(fill=tk.X)
        self.botao_selecionar_pasta = ttk.Button(
            linha_pasta,
            text="Selecionar pasta das peças...",
            command=self.selecionar_pasta
        )
        self.botao_selecionar_pasta.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(
            linha_pasta,
            textvariable=self.pasta_origem_var,
            anchor="w",
            justify="left",
            wraplength=1050
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Tabela: cada arquivo aparece apenas uma vez, com sua quantidade.
        linha_tabela = ttk.Frame(quadro_entrada)
        linha_tabela.pack(fill=tk.X, pady=(8, 0))
        quadro_tabela = ttk.Frame(linha_tabela)
        quadro_tabela.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tabela_arquivos = ttk.Treeview(
            quadro_tabela,
            columns=("arquivo", "quantidade"),
            show="headings",
            selectmode="extended",
            height=5
        )
        self.tabela_arquivos.heading("arquivo", text="Arquivo DXF")
        self.tabela_arquivos.heading("quantidade", text="Quantidade")
        self.tabela_arquivos.column("arquivo", width=650, anchor="w")
        self.tabela_arquivos.column("quantidade", width=110, anchor="center")
        barra_tabela = ttk.Scrollbar(
            quadro_tabela,
            orient=tk.VERTICAL,
            command=self.tabela_arquivos.yview
        )
        self.tabela_arquivos.configure(yscrollcommand=barra_tabela.set)
        self.tabela_arquivos.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        barra_tabela.pack(side=tk.RIGHT, fill=tk.Y)
        self.tabela_arquivos.bind("<<TreeviewSelect>>", self._ao_selecionar_linhas)

        botoes_lateral = ttk.Frame(linha_tabela)
        botoes_lateral.pack(side=tk.LEFT, padx=(10, 0), anchor="n")
        self.botao_todos = ttk.Button(
            botoes_lateral,
            text="Selecionar todos",
            command=self.selecionar_todos
        )
        self.botao_todos.pack(fill=tk.X, pady=(0, 5))
        self.botao_limpar = ttk.Button(
            botoes_lateral,
            text="Limpar seleção",
            command=self.limpar_selecao
        )
        self.botao_limpar.pack(fill=tk.X)

        linha_quantidade = ttk.Frame(quadro_entrada)
        linha_quantidade.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(
            linha_quantidade,
            text="Quantidade para as linhas selecionadas:"
        ).pack(side=tk.LEFT)
        self.campo_quantidade = ttk.Spinbox(
            linha_quantidade,
            from_=1,
            to=9999,
            increment=1,
            width=8,
            textvariable=self.quantidade_var
        )
        self.campo_quantidade.pack(side=tk.LEFT, padx=(8, 8))
        self.botao_aplicar_quantidade = ttk.Button(
            linha_quantidade,
            text="Aplicar quantidade",
            command=self.aplicar_quantidade
        )
        self.botao_aplicar_quantidade.pack(side=tk.LEFT)

        ttk.Label(
            quadro_entrada,
            textvariable=self.contagem_pecas_var,
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(6, 0))

        linha_controles = ttk.Frame(topo)
        linha_controles.pack(fill=tk.X)
        self.botao_iniciar = ttk.Button(
            linha_controles,
            text="Iniciar otimização",
            command=self.iniciar_otimizacao,
            state=tk.DISABLED
        )
        self.botao_iniciar.pack(side=tk.LEFT, padx=(0, 14))
        self.barra_progresso = ttk.Progressbar(
            linha_controles,
            mode="determinate",
            maximum=60,
            length=500
        )
        self.barra_progresso.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(
            linha_controles,
            textvariable=self.progresso_var,
            width=18,
            anchor="e"
        ).pack(side=tk.LEFT, padx=(12, 0))

        ttk.Label(topo, textvariable=self.status_var, font=("Segoe UI", 9)).pack(
            anchor="w", pady=(9, 0)
        )
        ttk.Label(
            topo,
            textvariable=self.resultado_var,
            font=("Segoe UI", 9, "bold"),
            wraplength=1250,
            justify="left"
        ).pack(anchor="w", pady=(4, 0))

        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X)
        area_graficos = ttk.Frame(self.root, padding=(12, 10))
        area_graficos.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.figura = Figure(figsize=(13, 6), dpi=100, constrained_layout=True)
        self.ax_chapa = self.figura.add_subplot(1, 2, 1)
        self.ax_convergencia = self.figura.add_subplot(1, 2, 2)
        self._mostrar_graficos_vazios()

        self.canvas = FigureCanvasTkAgg(self.figura, master=area_graficos)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    # --------------------------------------------------------------------------
    # Seleção dos arquivos e definição das quantidades
    # --------------------------------------------------------------------------
    def selecionar_pasta(self):
        pasta = filedialog.askdirectory(
            title="Selecione a pasta que contém as peças DXF",
            mustexist=True
        )
        if not pasta:
            return

        arquivos = sorted(
            os.path.join(pasta, nome)
            for nome in os.listdir(pasta)
            if nome.lower().endswith(".dxf")
            and os.path.isfile(os.path.join(pasta, nome))
        )

        self.pasta_origem = os.path.abspath(pasta)
        self.arquivos_dxf = arquivos
        self.quantidades_por_iid.clear()
        self.iid_para_arquivo.clear()
        for item in self.tabela_arquivos.get_children():
            self.tabela_arquivos.delete(item)

        self.pasta_origem_var.set(self.pasta_origem)
        for indice, caminho in enumerate(arquivos):
            iid = str(indice)
            self.quantidades_por_iid[iid] = 1
            self.iid_para_arquivo[iid] = caminho
            self.tabela_arquivos.insert(
                "",
                tk.END,
                iid=iid,
                values=(os.path.basename(caminho), 1)
            )

        if arquivos:
            self.contagem_pecas_var.set(
                f"{len(arquivos)} arquivo(s) DXF encontrado(s). "
                "Selecione os arquivos e defina quantas cópias de cada um deseja."
            )
            self.status_var.set("Selecione as peças que participarão da otimização.")
        else:
            self.contagem_pecas_var.set("Nenhum arquivo DXF foi encontrado nesta pasta.")
            self.status_var.set("Escolha uma pasta que contenha arquivos .dxf.")

        self._atualizar_selecao()

    def selecionar_todos(self):
        itens = self.tabela_arquivos.get_children()
        if itens:
            self.tabela_arquivos.selection_set(itens)
        self._atualizar_selecao()

    def limpar_selecao(self):
        self.tabela_arquivos.selection_remove(self.tabela_arquivos.selection())
        self._atualizar_selecao()

    def _ao_selecionar_linhas(self, _evento=None):
        selecionados = self.tabela_arquivos.selection()
        if len(selecionados) == 1:
            self.quantidade_var.set(str(self.quantidades_por_iid[selecionados[0]]))
        self._atualizar_selecao()

    def aplicar_quantidade(self):
        selecionados = self.tabela_arquivos.selection()
        if not selecionados:
            messagebox.showinfo(
                "Selecione os arquivos",
                "Selecione uma ou mais linhas antes de aplicar a quantidade."
            )
            return

        try:
            quantidade = int(self.quantidade_var.get())
            if quantidade < 1 or quantidade > 9999:
                raise ValueError
        except (TypeError, ValueError):
            messagebox.showwarning(
                "Quantidade inválida",
                "Informe um número inteiro entre 1 e 9999."
            )
            return

        for iid in selecionados:
            self.quantidades_por_iid[iid] = quantidade
            valores = list(self.tabela_arquivos.item(iid, "values"))
            self.tabela_arquivos.item(iid, values=(valores[0], quantidade))

        self._atualizar_selecao()

    def _obter_pecas_selecionadas(self):
        """Retorna pares (caminho DXF, quantidade) para as linhas selecionadas."""
        pecas = []
        for iid in self.tabela_arquivos.selection():
            caminho = self.iid_para_arquivo[iid]
            quantidade = self.quantidades_por_iid[iid]
            pecas.append((caminho, quantidade))
        return pecas

    def _atualizar_selecao(self):
        selecionados = self.tabela_arquivos.selection()
        quantidade_total = sum(
            self.quantidades_por_iid.get(iid, 0) for iid in selecionados
        )
        if self.arquivos_dxf:
            self.contagem_pecas_var.set(
                f"{len(self.arquivos_dxf)} DXF(s) na pasta | "
                f"{len(selecionados)} arquivo(s) selecionado(s) | "
                f"{quantidade_total} peça(s) no total."
            )

        habilitado = bool(selecionados) and quantidade_total > 0
        self.botao_iniciar.configure(state=tk.NORMAL if habilitado else tk.DISABLED)
        self.botao_aplicar_quantidade.configure(
            state=tk.NORMAL if selecionados else tk.DISABLED
        )

    def _habilitar_selecao(self, habilitar):
        estado = tk.NORMAL if habilitar else tk.DISABLED
        self.botao_selecionar_pasta.configure(state=estado)
        self.botao_todos.configure(state=estado)
        self.botao_limpar.configure(state=estado)
        self.tabela_arquivos.configure(selectmode="extended" if habilitar else "none")
        self.campo_quantidade.configure(state=estado)
        self.botao_aplicar_quantidade.configure(state=estado)

    # --------------------------------------------------------------------------
    # Gráficos e execução da otimização
    # --------------------------------------------------------------------------
    def _mostrar_graficos_vazios(self):
        self.ax_chapa.clear()
        self.ax_chapa.set_title("Distribuição das peças na chapa")
        self.ax_chapa.set_xlabel("Largura (mm)")
        self.ax_chapa.set_ylabel("Altura (mm)")
        self.ax_chapa.text(
            0.5, 0.5,
            "O layout será exibido após a otimização.",
            transform=self.ax_chapa.transAxes,
            ha="center", va="center"
        )

        self.ax_convergencia.clear()
        self.ax_convergencia.set_title("Convergência do algoritmo genético")
        self.ax_convergencia.set_xlabel("Geração")
        self.ax_convergencia.set_ylabel("Aproveitamento (%)")
        self.ax_convergencia.text(
            0.5, 0.5,
            "A curva será exibida após a otimização.",
            transform=self.ax_convergencia.transAxes,
            ha="center", va="center"
        )

    def iniciar_otimizacao(self):
        if self.thread_otimizacao is not None and self.thread_otimizacao.is_alive():
            return

        pecas_selecionadas = self._obter_pecas_selecionadas()
        quantidade_total = sum(qtd for _, qtd in pecas_selecionadas)
        if not pecas_selecionadas or quantidade_total == 0:
            messagebox.showwarning(
                "Nenhuma peça selecionada",
                "Selecione pelo menos um arquivo DXF com quantidade maior que zero."
            )
            return

        pasta_origem = self.pasta_origem
        self.botao_iniciar.configure(state=tk.DISABLED)
        self._habilitar_selecao(False)
        self.barra_progresso.configure(maximum=60, value=0)
        self.progresso_var.set("0 / 60 gerações")
        self.status_var.set(
            f"Preparando {quantidade_total} peça(s) a partir de "
            f"{len(pecas_selecionadas)} arquivo(s)..."
        )
        self.resultado_var.set("")
        self._mostrar_graficos_vazios()
        self.canvas.draw_idle()

        self.thread_otimizacao = threading.Thread(
            target=self._executar_otimizacao,
            args=(pecas_selecionadas, pasta_origem),
            daemon=True
        )
        self.thread_otimizacao.start()

    def _executar_otimizacao(self, arquivos_e_quantidades, pasta_origem):
        """Executa leitura, expansão das quantidades e algoritmo genético."""
        try:
            inicio = time.perf_counter()
            pasta_base = os.path.dirname(os.path.abspath(__file__))
            arquivo_saida = os.path.join(pasta_base, ARQUIVO_SAIDA_DXF)

            quantidade_total = sum(qtd for _, qtd in arquivos_e_quantidades)
            self.fila.put((
                "status",
                f"Lendo {len(arquivos_e_quantidades)} arquivo(s) e preparando "
                f"{quantidade_total} peça(s)..."
            ))

            perfis_originais = []
            nomes_pecas = []
            arquivos_origem_pecas = []

            # Cada arquivo é lido uma única vez. Sua geometria é repetida na
            # lista interna de peças conforme a quantidade definida na interface.
            for caminho_dxf, quantidade in arquivos_e_quantidades:
                poligono_base = carregar_poligono_de_dxf(caminho_dxf)
                nome_base = os.path.basename(caminho_dxf)

                for copia in range(quantidade):
                    perfis_originais.append(poligono_base)
                    if quantidade == 1:
                        nome_instancia = nome_base
                    else:
                        nome_instancia = f"{nome_base} [{copia + 1}]"
                    nomes_pecas.append(nome_instancia)
                    arquivos_origem_pecas.append(caminho_dxf)

            n_pecas = len(perfis_originais)
            if n_pecas == 0:
                raise RuntimeError("Nenhuma peça foi preparada para a otimização.")

            perfis_offset = [
                poligono.buffer(OFFSET_TOTAL, join_style=2)
                for poligono in perfis_originais
            ]

            def fitness_func(ga_instance, solution, solution_idx):
                ordem = np.argsort(solution[:n_pecas])
                indices_angulos = solution[n_pecas:]
                layout_off, layout_orig = decodificar_layout(
                    ordem,
                    indices_angulos,
                    perfis_offset,
                    perfis_originais
                )
                if layout_off is None:
                    return 0.0001

                x_max = max(poligono.bounds[2] for poligono in layout_off)
                area_pecas = sum(poligono.area for poligono in layout_orig)
                if x_max <= 0:
                    return 0.0001
                return area_pecas / (x_max * ALTURA_CHAPA)

            def acompanhar_geracao(ga_instance):
                geracao = ga_instance.generations_completed
                melhor = None
                if ga_instance.best_solutions_fitness:
                    melhor = float(ga_instance.best_solutions_fitness[-1])
                self.fila.put((
                    "progresso",
                    geracao,
                    ga_instance.num_generations,
                    melhor
                ))

            gene_space = (
                [{"low": 0.0, "high": 1.0}] * n_pecas
                + [list(range(len(ANGULOS_PERMITIDOS)))] * n_pecas
            )

            self.fila.put(("status", "Executando o algoritmo genético..."))
            ga = pygad.GA(
                num_generations=60,
                num_parents_mating=8,
                fitness_func=fitness_func,
                sol_per_pop=30,
                num_genes=2 * n_pecas,
                gene_space=gene_space,
                parent_selection_type="tournament",
                K_tournament=3,
                crossover_type="uniform",
                mutation_percent_genes=20,
                keep_elitism=2,
                on_generation=acompanhar_geracao
            )

            ga.run()
            melhor_solucao, melhor_fitness, _ = ga.best_solution()
            melhor_ordem = np.argsort(melhor_solucao[:n_pecas])
            melhores_angulos = melhor_solucao[n_pecas:]
            layout_off, layout_orig = decodificar_layout(
                melhor_ordem,
                melhores_angulos,
                perfis_offset,
                perfis_originais
            )

            if layout_off is None or layout_orig is None:
                raise RuntimeError(
                    "O algoritmo terminou, mas não encontrou um layout viável. "
                    "Tente reduzir as quantidades, verificar as dimensões das peças "
                    "ou alterar os parâmetros de otimização."
                )

            arquivo_saida = exportar_layout_para_dxf(
                layout_orig,
                LARGURA_CHAPA,
                ALTURA_CHAPA,
                arquivo_saida
            )

            tempo_total = time.perf_counter() - inicio
            self.fila.put((
                "concluido",
                {
                    "layout_off": layout_off,
                    "layout_orig": layout_orig,
                    "ordem": melhor_ordem,
                    "nomes_pecas": nomes_pecas,
                    "historico": list(ga.best_solutions_fitness),
                    "fitness": float(melhor_fitness),
                    "tempo": tempo_total,
                    "arquivo_saida": arquivo_saida,
                    "pasta_origem": pasta_origem,
                    "n_pecas": n_pecas,
                    "n_arquivos": len(arquivos_e_quantidades),
                }
            ))

        except Exception:
            self.fila.put(("erro", traceback.format_exc()))

    def _processar_fila(self):
        try:
            while True:
                mensagem = self.fila.get_nowait()
                tipo = mensagem[0]

                if tipo == "status":
                    self.status_var.set(mensagem[1])

                elif tipo == "progresso":
                    geracao, total, melhor = mensagem[1:]
                    self.barra_progresso.configure(maximum=total, value=geracao)
                    self.progresso_var.set(f"{geracao} / {total} gerações")
                    if melhor is not None:
                        self.status_var.set(
                            "Otimizando... melhor aproveitamento atual: "
                            f"{melhor * 100:.2f}%"
                        )

                elif tipo == "concluido":
                    self._finalizar_com_sucesso(mensagem[1])

                elif tipo == "erro":
                    self.thread_otimizacao = None
                    self._habilitar_selecao(True)
                    self._atualizar_selecao()
                    self.status_var.set("A otimização não foi concluída.")
                    messagebox.showerror("Erro durante a otimização", mensagem[1])

        except queue.Empty:
            pass

        self.root.after(100, self._processar_fila)

    def _finalizar_com_sucesso(self, resultado):
        self.thread_otimizacao = None
        self._habilitar_selecao(True)
        self._atualizar_selecao()
        self.barra_progresso.configure(maximum=60, value=60)
        self.progresso_var.set("60 / 60 gerações")
        self.status_var.set("Otimização concluída.")
        self.resultado_var.set(
            f"Arquivos selecionados: {resultado['n_arquivos']} | "
            f"Peças posicionadas: {resultado['n_pecas']} | "
            f"Aproveitamento: {resultado['fitness'] * 100:.2f}% | "
            f"Tempo: {resultado['tempo']:.2f} s | "
            f"DXF salvo em: {resultado['arquivo_saida']}"
        )
        self._desenhar_resultados(resultado)

    def _desenhar_resultados(self, resultado):
        ax = self.ax_chapa
        ax.clear()
        cx, cy = chapa.exterior.xy
        ax.fill(cx, cy, color="#f4f4f4", zorder=0)
        ax.plot(cx, cy, color="black", linewidth=2, label="Limite da chapa")

        cores = plt.cm.tab10.colors
        for i, (p_off, p_orig) in enumerate(zip(
            resultado["layout_off"],
            resultado["layout_orig"]
        )):
            cor = cores[i % len(cores)]
            ox, oy = p_off.exterior.xy
            ax.plot(ox, oy, linestyle="--", color="gray", linewidth=0.8, zorder=1)

            px, py = p_orig.exterior.xy
            ax.fill(
                px, py,
                alpha=0.7,
                facecolor=cor,
                edgecolor="black",
                linewidth=1.2,
                zorder=2
            )

            for interior in p_orig.interiors:
                ix, iy = interior.xy
                ax.fill(ix, iy, facecolor="#f4f4f4", edgecolor="black", linewidth=0.8)

            ax.text(
                p_orig.centroid.x,
                p_orig.centroid.y,
                resultado["nomes_pecas"][int(resultado["ordem"][i])],
                fontsize=7,
                ha="center",
                va="center",
                fontweight="bold"
            )

        x_corte_final = max(poligono.bounds[2] for poligono in resultado["layout_off"])
        ax.axvline(
            x_corte_final,
            color="red",
            linestyle=":",
            linewidth=1.5,
            label=f"Fim das peças (X={x_corte_final:.0f} mm)"
        )
        ax.set_title(
            f"Layout otimizado — aproveitamento: {resultado['fitness'] * 100:.1f}%"
        )
        ax.set_xlabel("Largura (mm)")
        ax.set_ylabel("Altura (mm)")
        ax.set_xlim(0, LARGURA_CHAPA)
        ax.set_ylim(0, ALTURA_CHAPA)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="upper right", fontsize=8)

        ax = self.ax_convergencia
        ax.clear()
        historico = np.asarray(resultado["historico"], dtype=float) * 100
        ax.plot(
            np.arange(1, len(historico) + 1),
            historico,
            color="#1f77b4",
            linewidth=2
        )
        ax.set_title("Convergência do algoritmo genético")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Aproveitamento (%)")
        ax.set_xlim(left=1)
        ax.grid(True, linestyle="--", alpha=0.5)
        self.canvas.draw_idle()


# ==============================================================================
# 4. INICIALIZAÇÃO DA APLICAÇÃO
# ==============================================================================
if __name__ == "__main__":
    janela = tk.Tk()
    aplicacao = AplicacaoNesting(janela)
    janela.mainloop()

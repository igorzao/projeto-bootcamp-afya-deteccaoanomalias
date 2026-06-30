import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os


def carregar_dados(caminho_arquivo):
    """
    Carrega a base de transações a partir de um arquivo CSV.
    """
    df = pd.read_csv(caminho_arquivo)
    return df


def tratar_dados(df):
    """
    Realiza tratamentos básicos na base de dados.
    """
    df = df.copy()

    df = df.dropna()

    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

    df = df.dropna(subset=["valor"])

    return df


def detectar_anomalias(df):
    """
    Aplica o modelo Isolation Forest para identificar transações anômalas.
    """
    df = df.copy()

    dados_modelo = df[["valor"]]

    modelo = IsolationForest(
        contamination=0.15,
        random_state=42
    )

    df["resultado_modelo"] = modelo.fit_predict(dados_modelo)

    df["classificacao"] = df["resultado_modelo"].apply(
        lambda x: "Anomalia" if x == -1 else "Normal"
    )

    df = df.drop(columns=["resultado_modelo"])

    return df


def gerar_saida(df, caminho_saida):
    """
    Gera o arquivo final com as transações classificadas.
    """
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    df.to_csv(caminho_saida, index=False, encoding="utf-8-sig")


def exibir_resumo(df):
    """
    Exibe um resumo da classificação das transações.
    """
    print("Resumo da detecção de anomalias:")
    print(df["classificacao"].value_counts())
    print("\nTransações classificadas como anomalia:")
    print(df[df["classificacao"] == "Anomalia"])


def main():
    caminho_entrada = "data/transacoes.csv"
    caminho_saida = "outputs/transacoes_classificadas.csv"

    df = carregar_dados(caminho_entrada)

    df_tratado = tratar_dados(df)

    df_classificado = detectar_anomalias(df_tratado)

    gerar_saida(df_classificado, caminho_saida)

    exibir_resumo(df_classificado)

    print(f"\nArquivo gerado com sucesso em: {caminho_saida}")


if __name__ == "__main__":
    main()

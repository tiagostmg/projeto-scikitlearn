from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder # Importar LabelEncoder
import pandas as pd
import numpy as np # Importar numpy para tratar NaNs

# --- 1. Carregar os Dados ---
# Carregar o arquivo de treino
df_train = pd.read_csv("data/train_u6lujuX_CVtuZ9i.csv")
# Carregar o arquivo de teste (este será usado para a submissão final)
df_test = pd.read_csv("data/test_Y3wMUE5_7gLdaTN.csv") # Certifique-se de que o nome do arquivo está correto

# Guardar o Loan_ID do arquivo de teste para a submissão
test_loan_ids = df_test['Loan_ID']

# --- 2. Pré-processamento (Aplicar em ambos os datasets) ---

# Função para pré-processamento para garantir consistência
def preprocess_data(df, label_encoders=None, is_train=True):
    # 2.1. Preencher valores nulos
    # Uma estratégia melhor que ffill para datasets mistos é preencher com moda para categóricas
    # e mediana/média para numéricas.
    # Para simplificar agora, vamos usar ffill e depois uma média/moda,
    # mas considere estratégias mais robustas para dados reais.

    # Preencher colunas numéricas com a mediana
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    # Preencher colunas categóricas com a moda (valor mais frequente)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 2.2. Codificar colunas categóricas
    if label_encoders is None: # Se não houver encoders (primeira vez, no treino)
        label_encoders = {}
        for column in df.columns:
            if df[column].dtype == 'object':
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column].astype(str))
                label_encoders[column] = le
    else: # Se houver encoders (usar os do treino para o teste)
        for column, le in label_encoders.items():
            if column in df.columns: # Apenas para colunas presentes no dataset
                # Lidar com categorias desconhecidas no teste:
                # Se uma categoria no teste não foi vista no treino,
                # LabelEncoder irá gerar um erro. Podemos tratá-las como NaN ou uma nova categoria.
                # Para simplicidade, vamos usar 'transform', mas em um cenário real,
                # você pode querer mapear categorias desconhecidas para um valor específico
                # ou usar OneHotEncoder que lida melhor com isso.
                df[column] = df[column].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1) # -1 para desconhecidos
                # É importante notar que LabelEncoder não lida bem com categorias "novas" no teste.
                # Para uma solução mais robusta, especialmente com muitos valores,
                # OneHotEncoder com handle_unknown='ignore' seria melhor.

    return df, label_encoders

# Pré-processar dados de treino e obter os encoders
df_train_processed, label_encoders = preprocess_data(df_train.copy(), is_train=True)

# Pré-processar dados de teste usando os encoders do treino
df_test_processed, _ = preprocess_data(df_test.copy(), label_encoders=label_encoders, is_train=False)

# Remover 'Loan_ID' dos datasets processados, pois não é uma feature para o modelo
df_train_processed = df_train_processed.drop(columns=['Loan_ID'])
df_test_processed = df_test_processed.drop(columns=['Loan_ID'])

# --- 3. Separar X e y para o Treino ---
X = df_train_processed.drop(columns=["Loan_Status"])
y = df_train_processed["Loan_Status"]

# --- 4. Dividir entre Treino e Validação (do arquivo de treino) ---
# Esta divisão é para você avaliar a performance do seu modelo internamente
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Treinar Modelo ---
model = RandomForestClassifier(random_state=42) # Adicione random_state para reprodutibilidade
model.fit(X_train, y_train)

# --- 6. Avaliar o Modelo (Opcional, mas recomendado) ---
y_pred_val = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_val)
print(f"Acurácia no conjunto de validação: {accuracy:.4f}")

# --- 7. Fazer Previsões no Arquivo de Teste Real ---
# Fazer previsões no df_test_processed (que veio do test_Y3wMUE5_7gLwTgJ.csv)
predictions = model.predict(df_test_processed)

# As previsões (`predictions`) estão em formato numérico (0 ou 1)
# O Kaggle geralmente espera 'Y' ou 'N' para Loan_Status.
# Vamos decodificar as previsões de volta para 'Y' e 'N' usando o LabelEncoder de 'Loan_Status'.
# Primeiro, encontre o encoder para 'Loan_Status'
loan_status_le = label_encoders['Loan_Status']

# Mapear 0 para 'N' e 1 para 'Y' (assumindo que 'N' foi codificado como 0 e 'Y' como 1)
# Se não tiver certeza, você pode imprimir loan_status_le.classes_
# Ex: print(loan_status_le.classes_) # Saída esperada: ['N', 'Y'] se N=0, Y=1
decoded_predictions = loan_status_le.inverse_transform(predictions)


# --- 8. Criar o Arquivo de Submissão ---
submission_df = pd.DataFrame({
    'Loan_ID': test_loan_ids,
    'Loan_Status': decoded_predictions
})

# Salvar o arquivo de submissão
submission_df.to_csv('submission.csv', index=False)

print("\nArquivo 'submission.csv' gerado com sucesso!")
print(submission_df.head())
FROM python:3.9-slim

WORKDIR /app

# Instala ferramentas básicas do Linux (necessárias para compilar algumas libs)
RUN apt-get update && apt-get install -y build-essential curl

# Copia as dependências e instala
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copia o código do App
COPY . .

# Comando que roda quando o container liga
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
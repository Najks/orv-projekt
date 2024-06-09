# Uporabimo uradno Python 3.8 osnovno sliko
FROM python:3.8-slim

# Nastavimo delovni imenik
WORKDIR /app

# Namestimo odvisnosti, ki jih zahteva opencv
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Kopiramo requirements.txt v delovni imenik in namestimo Python odvisnosti
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Kopiramo celotno aplikacijo v delovni imenik
COPY . /app

# Ustvarimo potrebne mape
RUN mkdir -p /app/uploads /app/frames /app/processed /app/augmented

# Nastavimo privzeti port za aplikacijo
ENV PORT 5000

# Expose the port the app runs on
EXPOSE $PORT

# Definiramo privzeti ukaz za zagon aplikacije
CMD ["python", "app.py"]

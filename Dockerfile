# Uporabimo uradno Python 3.8 osnovno sliko
FROM python:3.8-slim

# Nastavimo delovni imenik
WORKDIR /app

# Kopiramo requirements.txt v delovni imenik in namestimo odvisnosti
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Kopiramo celotno aplikacijo v delovni imenik
COPY . /app

# Ustvarimo potrebne mape
RUN mkdir -p /app/uploads /app/frames /app/processed /app/augmented

# Expose the port the app runs on
EXPOSE 5000

# Definiramo privzeti ukaz za zagon aplikacije
CMD ["python", "app.py"]

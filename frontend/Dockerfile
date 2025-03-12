# Étape 1 : Utiliser une image officielle de Node.js
FROM node:21

# Créer le répertoire de travail dans le conteneur
WORKDIR /app

# Installer Create React App globalement
RUN npm install -g create-react-app

# Créer un projet React dans le conteneur
RUN npx create-react-app my-react-app

# Se placer dans le répertoire du projet
WORKDIR /app/my-react-app
RUN npm install axios
RUN npm install react-sketch
RUN npm install react-canvas-draw --legacy-peer-deps
RUN npm install --legacy-peer-deps
COPY . .
# VOLUME [ "/app/src" ]
# That syntax won't work in a Dockerfile. 
# When using VOLUME, you can only specify the container path (e.g. VOLUME ["/app/my-react-app"]).
# Host paths (like '.') and read-only flags should be set at runtime with -v or in docker-compose.
# Exposer le port 3000
EXPOSE 3000

# Démarrer l'application React en mode développement
CMD ["npm", "start"]


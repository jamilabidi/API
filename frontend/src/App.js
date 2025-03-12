import React, { useState } from 'react';
import axios from 'axios';
import CanvasDraw from "react-canvas-draw";

const DrawingCanvas = () => {
    const [canvas, setCanvas] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [prediction, setPrediction] = useState(""); // Stocker la prédiction
    const [userNumber, setUserNumber] = useState(""); // Stocker le nombre dessiné par l'utilisateur

    const handleSave = async () => {
        if (!canvas) return;

        const image = canvas.getDataURL(); // Image en base64
        setIsLoading(true);
        setPrediction(""); // Réinitialiser l'affichage de la prédiction

        try {
            const response = await axios.post('http://localhost:8000/api/predict', { image });
            console.log('Réponse du serveur:', response.data);
            
            // Mettre à jour la prédiction
            setPrediction(response.data.prediction);
        } catch (error) {
            console.error('Erreur lors de l\'envoi:', error);
            setPrediction("Erreur lors de la prédiction !");
        } finally {
            setIsLoading(false);
        }
    };


    
    // Appel de la fonction au chargement de la page
    fetchDebugImages();
    const handleSubmit = async (event) => {
        event.preventDefault();
    
        if (!canvas || userNumber === "") {
            alert("Veuillez entrer un nombre et dessiner quelque chose avant d'envoyer.");
            return;
        }
    
        const image = canvas.getDataURL(); // Image en base64
    
        try {
            const response = await axios.post(
                'http://localhost:8000/api/helpdata',
                { image, number: userNumber },
                { headers: { "Content-Type": "application/json" } } // Ajout des headers
            );
    
            console.log('Données envoyées:', response.data);
            alert(`Donnée envoyée avec succès ! Fichier: ${response.data.filename}`);
        } catch (error) {
            console.error("Erreur lors de l'envoi des données :", error);
            alert("Erreur lors de l'envoi !");
        }
    };
    
    async function fetchDebugImages() {
        try {
            const response = await fetch("http://localhost:8000/api/debug_images");
            const data = await response.json();
    
            const container = document.getElementById("debug-images-container");
            container.innerHTML = ""; // Effacer les anciennes images
            // recupérer la derniere image
            const imgElement = document.createElement("img");
            imgElement.src = data.images[data.images.length - 1];
            imgElement.alt = "Image de debug";
            imgElement.style.width="200px";
            container.appendChild(imgElement);
        } catch (error) {
            console.error("Erreur lors de la récupération des images :", error);
        }
    }


    return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "20px" }}>
            <h1>Prédiction avec dessin</h1>

            <CanvasDraw
                ref={(canvasDraw) => setCanvas(canvasDraw)}
                brushColor="black"
                brushRadius={15}
                canvasWidth={300}
                canvasHeight={300}
                style={{ border: "2px solid black" }}
            />

            <button onClick={handleSave} disabled={isLoading} style={{ marginTop: "10px" }}>
                {isLoading ? 'Envoi en cours...' : 'Envoyer le dessin'}
            </button>

            <button onClick={() => canvas.clear()} style={{ marginTop: "10px" }}>
                Effacer le dessin
            </button>

            {prediction && (
                <div style={{ marginTop: "20px", fontSize: "18px", fontWeight: "bold" }}>
                    Prédiction : {prediction}
                </div>
            )}

            {/* Formulaire pour envoyer l'image et le nombre dessiné */}
            <form onSubmit={handleSubmit} style={{ marginTop: "20px", display: "flex", flexDirection: "column", alignItems: "center" }}>
                <label>
                    Entrez le nombre dessiné :
                    <input
                        type="number"
                        value={userNumber}
                        onChange={(e) => setUserNumber(e.target.value)}
                        required
                        style={{ marginLeft: "10px", padding: "5px" }}
                    />
                </label>

                <button type="submit" style={{ marginTop: "10px" }}>
                    Envoyer pour aide
                </button>
            </form>
            <div>
                <h3>Images de Debug</h3>
                <div id="debug-images-container"></div>
            </div>
        </div>
    );
};

export default DrawingCanvas;



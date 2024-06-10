import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

# Chemin vers votre image
image_path = 'C:\\Users\\hiche\\Desktop\\dormir.jpg'

# Tableau des textes possibles
textes = ["Le bébé dort"]

try:
    # Charger l'image avec Pillow
    image = Image.open(image_path)

    # Convertir l'image en array numpy pour utilisation avec Matplotlib
    image_np = np.array(image)

    # Créer la figure et les axes
    fig, ax = plt.subplots()
    ax.imshow(image_np)  # Afficher l'image
    ax.axis('off')  # Désactiver les axes pour un affichage clair

    # Sélectionner un texte aléatoire
    texte_aleatoire = random.choice(textes)

    # Ajouter un peu d'espace au bas de la figure pour le texte
    fig.subplots_adjust(bottom=0.2)

    # Ajouter le texte sous l'image
    # Placer le texte au milieu de la figure, sous l'image
    fig.text(0.5, 0.1, texte_aleatoire, ha='center', fontsize=12)

    # Afficher l'image modifiée
    plt.show()

except FileNotFoundError:
    print(f"Le fichier à l'emplacement '{image_path}' n'a pas été trouvé.")
except Exception as e:
    print(f"Une erreur s'est produite : {e}")

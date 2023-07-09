# Exploring the Underlying Emotional Models in Emotion Recognition Systems with Electrodermal Activity
Repository of the article **'Exploring the Underlying Emotional Models in Emotion Recognition Systems with Electrodermal Activity'**
* APA citation: 
* Pre-registration available at: https://osf.io/ewuaf

# Rationale
Affective computing is an interdisciplinary field that combines computer science and engineering to automatically recognize and interpret emotions. Recent research has focused on using physiological signals (e.g., electrodermal activity) to improve emotion recognition. However, little attention has been paid to the theoretical emotion models underlying these systems. Thus, we conducted a systematic review and meta-analysis of the literature on automatic emotion recognition systems using electrodermal activity. We found a mismatch between the types of machine learning models and the emotional models used. Furthermore, we found that arousal prediction models consistently outperform valence prediction models, consistent with affective science evidence. We conclude that a comprehensive understanding of affective states requires consideration of both affective and computational perspectives in emotion research.

Keywords: affective computing, emotion recognition, electrodermal activity, emotional models, systematic review, meta-analysis 

# Structure
Repository
│
└───data
│    │
│    └───cleaned
│    │   │
│    │   └───
│    │
│    └───processed
│    │   │
│    │   └───
│    │
│    └───raw
│        │
│        └───
└───figures
│    │
│    └───plots
│    │
│    └───tableau
│    
└───notebooks
│    │
│    └───data_analysis.ipynb
│    
└───scripts
│    │
│    └───metaanalysis.py
│    
│   README.md
│   AUTHORS.md
│   CITATION.cff
│   requirements.txt

# Guide
Para replicar los resultados reportados por esta review, se sugiere seguir los siguientes pasos:
* Siguiendo el flowchart, los trabajos incluidos al principio de esta review estan ACA, y los trabajos incluidos al final estan ACA. Importar para descubrir mas.
* Las tablas en AQUI fueron desarrolladas por los investigadores, con la lectura de estos trabajos. 
* De estas tablas se desprendieron las de ACA, con las cuales se exejuto el analisis de datos.
* Setear el enviroment:
 python<version> -m venv <virtual-environment-name>
!source venv/bin/activate
!pip install -r requirements.txt
* 
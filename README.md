# Exploring the Underlying Emotional Models in Emotion Recognition Systems with Electrodermal Activity
Repository of the article **'Exploring the Underlying Emotional Models in Emotion Recognition Systems with Electrodermal Activity'**
* APA citation: 
* Pre-registration available at: https://osf.io/ewuaf
* Authors
    - Tomas A. D'Amelio²³ (https://github.com/tomdamelio, dameliotomas@gmail.com)
    - Lorenzo A. Galán¹ (https://github.com/LorenzoGalan, lorenzogalan43@gmail.com) 
    - Emmanuel A. Maldonado¹ (https://github.com/EmmAMaldonado, emmanuel.a.maldonado@protonmail.com)
    - Agustin A. Diaz Barquinero¹ (https://github.com/agusdiazb, agustindiazbarquinero@gmail.com) 

¹Department of Psychology, University of Buenos Aires, Buenos Aires, Argentina.
²Department of Physics, University of Buenos Aires, Buenos Aires, Argentina.
³National Scientific and Technical Research Council (CONICET), Buenos Aires, Argentina.

Todas las consultas relacionadas a este trabajo deberian ser enviadas dameliotomas@gmail.com

# Rationale
Affective computing is an interdisciplinary field that combines computer science and engineering to automatically recognize and interpret emotions. Recent research has focused on using physiological signals (e.g., electrodermal activity) to improve emotion recognition. However, little attention has been paid to the theoretical emotion models underlying these systems. Thus, we conducted a systematic review and meta-analysis of the literature on automatic emotion recognition systems using electrodermal activity. We found a mismatch between the types of machine learning models and the emotional models used. Furthermore, we found that arousal prediction models consistently outperform valence prediction models, consistent with affective science evidence. We conclude that a comprehensive understanding of affective states requires consideration of both affective and computational perspectives in emotion research.

Keywords: affective computing, emotion recognition, electrodermal activity, emotional models, systematic review, meta-analysis 

# Structure
Repository
- data: after defining the papers to be analyzed, a general table was created where all the papers and the characteristics of each one to be studied were written down. This folder contains the general table in excel format (intended to be visualized quickly and conveniently). From this, x sheets were extracted, each one being a subcategory of the characteristics to be studied (i.e.: metadata collects the titles, authors, type of article, etc.), which are in csv format, for the creation of the dataframes to be used in the data analysis. In the subfolder x is the POWERBI file for the creation of the world map.
- figures: the figures used in the publication of this work are in this folder. They were created with the script x located in the scripts folder.
- notebooks: the notebook used for data exploration and analysis is located in this folder.
- scripts: the scrpit for the creation of the plots of this article.
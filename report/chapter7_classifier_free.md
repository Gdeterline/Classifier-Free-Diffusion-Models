\chapter{Classifier-Free Guidance}
\label{chap:classifier_free}
\rhead{Classifier-Free Guidance}

\section{Motivations}

Dans le chapitre précédent, nous avons détaillé comment la Classifier Guidance permet de contrôler le processus de génération des modèles de diffusion en utilisant le gradient issu d'un modèle de classification. Bien que cette méthode soit efficace, elle s'accompagne de contraintes majeures qui limitent son application pratique.\\

La principale difficulté réside dans la nécessité de concevoir, d'entraîner et d'intégrer un réseau de neurones externe  supplémentaire. Ce classifieur doit être suffisamment robuste pour classifier des images soumises à des niveaux de bruit qui peuvent être extrêmes (pour des valeurs des pas de temps $t$ proches de $T$), afin de correspondre aux états intermédiaires $x_t$ du processus de diffusion.\\

Pour pallier à ces limitations, à la fois architecturales et pratiques, Jonathan Ho et Tim Salimans ont introduit en 2021 la \textbf{Classifier-Free Guidance} \cite{ho2022}. L'objectif de cette approche est de permettre un conditionnement du processus de génération sans pour autant nécessiter de modèle de classification externe.\\

\underline{Note:} Dans toute la suite, nous considérerons un problème de classification, où $y$ représente une classe. La méthode reste applicable à bien d'autres types de conditions (texte, attributs, etc.).\\

\section{Principe général du Guidage sans classifieur (Classifier-Free Guidance)}

Contrairement à la Classifier Guidance, qui s'appuie sur un modèle de classification pour guider la génération, le guidage sans classifieur (CFG) repose sur l'idée d'entraîner un modèle de diffusion conditionnel qui prend en entrée non seulement une image bruitée $x_t$ et le pas de temps $t$, mais aussi une classe $y$. L'objectif est d'entraîner le modèle sur des exemples conditionnels (où la classe/condition $y$ associée est fournie), mais aussi sur des exemples inconditionnels (où la classe/condition est remplacée par un token nul), afin de permettre au modèle d'apprendre à générer des images à la fois avec et sans conditionnement.\\

Dans cette section, nous présentons les fondements de la Classifier-Free Guidance, en détaillant dans un premier temps sa formulation mathématique, puis en proposant une intuition géométrique pour mieux comprendre son fonctionnement (basé sur les travaux S. Dieleman \cite{dieleman2023}).\\

\subsection{Formulation mathématique de la CFG (Classifier-Free Guidance/Guidage sans classifieur)}

L'idée fondamentale du guidage sans classifieur repose sur une reformulation algébrique du théorème de Bayes appliqué aux fonctions de score.\\

Si nous reprenons la formulation de la Classifier Guidance (équation \ref{eq:cg_formulation_finale}), nous avons :\\

\begin{equation}
\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(x_t) + s \nabla_{x_t} \log p(y | x_t)
\label{eq:cg_initial}
\end{equation}

\vspace{0.3cm}

où $\nabla_{x_t} \log p(y | x_t)$ est le gradient de la probabilité que l'image $x_t$ appartienne à la classe $y$, et qui est estimé à l'aide d'un classifieur externe, et $s$ est le facteur d'échelle qui contrôle l'intensité du conditionnement.\\

De même que nous l'avons fait pour la Classifier Guidance (voir équation \ref{eq:cg_bayes}), nous partons du théorème de Bayes, mais en l'appliquant cette fois-ci à la probabilité conditionnelle $p(y | x_t)$ (et non plus à $p(x_t | y)$).
Nous avons donc:\\

\begin{equation}
\nabla_{x_t} \log p(y | x_t) = \nabla_{x_t} \log p(x_t | y) - \nabla_{x_t} \log p(x_t)
\label{eq:cfg_bayes}
\end{equation}

\vspace{0.3cm}

Cette formule est cruciale, dans la mesure où elle nous permet d'obtenir, simplement en soustrayant le score inconditionnel du modèle conditionnel, le score d'un classifieur externe. \\

Nous pouvons réécrire l'expression \ref{eq:cg_initial} de la manière suivante : \\

\begin{equation}
\nabla_{x_t} \log p(x_t | y) = (1 - s) \nabla_{x_t} \log p(x_t) + s \nabla_{x_t} \log p(x_t | y)
\label{eq:cfg_combined}
\end{equation}

\vspace{0.3cm}

Nous avons alors une combinaison linéaire entre le score inconditionnel $\nabla_{x_t} \log p(x_t)$ et le score conditionnel $\nabla_{x_t} \log p(x_t | y)$, pondérée par le paramètre $s$, qui nous permet de conditionner le processus de génération en s'affranchissant d'un classifieur externe.\\

Lors de l'entraînement du modèle, nous présentons à la fois des exemples conditionnels (où la classe $y$ est donnée) et des exemples inconditionnels (où la classe est remplacée par un token nul, on parle de "dropout" de classe). Le modèle de diffusion est alors capable de générer des images à la fois avec et sans conditionnement. Lors de l'inférence, à chaque pas de temps, nous pouvons alors calculer les deux scores (conditionnel et inconditionnel) et les combiner pour guider la génération selon le niveau de guidage souhaité. \\

Selon les valeurs de $s$, nous pouvons obtenir différents comportements de génération :
\begin{itemize}
    \item $s = 0$ : Génération purement inconditionnelle, où le modèle génère des images sans tenir compte de la condition $y$.
    \item $s = 1$ : Génération purement conditionnelle, où le modèle génère des images strictement conformes à la condition $y$.
    \item $s > 1$ : Génération avec un guidage renforcé, où le modèle est fortement incité à suivre la condition $y$, au risque de réduire la diversité des échantillons générés. \\
\end{itemize}

Afin de mieux comprendre les mécanismes sous-jacents de cette formulation mathématique, nous proposons désormais une approche géométrique complémentaire de ce qui a été proposé.\\

\subsection{Intuition géométrique de la CFG}

Dans cette section, nous présentons une intuition géométrique de la CFG, qui devrait illustrer de manière plus visuelle comment la combinaison des scores conditionnel et inconditionnel permet de guider le processus de génération dans l'espace des images.
Considérons une image bruitée $x_t$ à un pas de temps donné $t$. Dans l'espace des images, nous pouvons imaginer que les différentes classes forment des régions distinctes. \\

Considérons une image bruitée $x_t$ à un pas de temps donné $t$. Dans l'espace des images, nous pouvons imaginer que les différentes classes forment des régions distinctes. \\

À l'instant $t$, le modèle de diffusion effectue deux prédictions: celle de $\hat{x}_0$ inconditionnelle (sans conditionnement) et celle de $\hat{x}_0 | y$ conditionnelle. Ces deux prédictions peuvent être représentées comme des points dans l'espace des images. La figure \ref{fig:cfg_predictions} illustre cette situation, où nous avons une prédiction inconditionnelle et une prédiction conditionnelle. \\

La formule \ref{eq:cfg_bayes} nous indique que le score de la CFG est une combinaison linéaire entre le score inconditionnel et le score conditionnel. Géométriquement, cela signifie que nous pouvons représenter ce score comme un vecteur $\delta$ caractérisant l'écart entre la prédiction inconditionnelle et la prédiction conditionnelle, comme illustré dans la figure \ref{fig:cfg_delta}. \\

En multipliant ce vecteur $\delta$ par un facteur de guidage $s$, nous pouvons ajuster l'intensité de ce vecteur, ce qui revient à renforcer ou atténuer le guidage vers la condition $y$. La figure \ref{fig:cfg_scale_delta} montre comment le vecteur $\delta$ est mis à l'échelle par le paramètre $s$. \\

Ensuite, lors de la mise à jour de $x_t$ pour obtenir $x_{t-1}$, nous ajoutons ce vecteur de score à la prédiction inconditionnelle, ce qui nous permet de guider la génération vers la condition souhaitée. La figure \ref{fig:cfg_step} illustre cette étape, où le vecteur de score est utilisé pour ajuster la prédiction inconditionnelle. \\

Enfin, l'ajout de bruit pour obtenir $x_{t-1}$ à partir de la prédiction ajustée est représenté dans la figure \ref{fig:cfg_add_noise}. Cette étape est très importante dans la génération. En effet, si elle n'avait pas lieu, la génération deviendrait alors déterministe, et nous perdrions la diversité des échantillons générés. L'ajout de bruit permet de maintenir une certaine variabilité dans les échantillons, même lorsque le guidage est fort. Notons que nous n'ajoutons pas de bruit lors de la dernière étape de génération (lorsque $t=1$), afin d'obtenir une image finale nette. \\

\begin{figure}[htbp]
    \centering
    % Première ligne : 3 images
    \begin{subfigure}[b]{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/cfg_geometry_predict_cond.png}
        \caption{Prédiction inconditionnelle et conditionnelle}
        \label{fig:cfg_predictions}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/cfg_geometry_delta.png}
        \caption{Vecteur $\delta$ entre les prédictions}
        \label{fig:cfg_delta}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/cfg_geometry_scale_delta.png}
        \caption{Vecteur $\delta$ mis à l'échelle par $s$}
        \label{fig:cfg_scale_delta}
    \end{subfigure}

    \vspace{1em} % Espace vertical entre les deux lignes

    % Deuxième ligne : 2 images, centrées
    \begin{subfigure}[b]{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/cfg_geometry_step.png}
        \caption{Vector de score utilisé pour la mise à jour de $x_t$}
        \label{fig:cfg_step}
    \end{subfigure}
    \hspace{2em} % Espace horizontal entre les deux images du bas
    \begin{subfigure}[b]{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/cfg_geometry_add_noise.png}
        \caption{Ajout de bruit pour obtenir $x_{t-1}$}
        \label{fig:cfg_add_noise}
    \end{subfigure}

    \caption{Représentation géométrique simplifiée de la Classifier-Free Guidance \cite{dieleman2023}.}
    \label{fig:cfg_geometry}
\end{figure}

Nous avons ainsi une intuition géométrique de la CFG, qui nous permet de visualiser comment les différentes composantes du score interagissent pour guider le processus de génération dans l'espace des images. Cette perspective peut être très utile pour comprendre les effets du paramètre de guidage $s$ et pour interpréter les résultats expérimentaux que nous présenterons par la suite. \\

\underline{Note:} Les figures \ref{fig:cfg_predictions}, \ref{fig:cfg_delta}, \ref{fig:cfg_scale_delta}, et \ref{fig:cfg_add_noise} montrent des espaces en 2 dimensions pour illustrer les différentes composantes du score et leur combinaison. Il est important de noter que ces figures sont ici pour aider à la compréhension, mais que l'espace réel des images est de beaucoup plus haute dimension.

\section{Application aux DDPMs}

Ayant présenté et donné une intuition géométrique de la Classifier-Free Guidance, nous pouvons maintenant appliquer cette méthode aux modèles de diffusion basés sur les DDPMs. 

\subsection{Intégration des Pas de Temps et des Classes}

En termes d'implémentation, le DDPM conditionnel n'est pas très différent du DDPM inconditionnel. La principale différence par rapport à l'implémentation inconditionnelle réside dans la prise en compte du pas de temps $t$ et de la classe $y$ (ou de la condition $y$ plus généralement) dans l'architecture du modèle. Les deux types d'informations (pas de temps et classe) sont encodées et injectées dans les blocks de ResNet du modèle de diffusion, au sein des différentes "couches" du modèle. \\

\subsubsection{Encodage du Pas de Temps}

L'encodage du pas de temps se fait de la même façon que pour l'implémentation du DDPM présentée dans le chapitre 3. Nous utilisons une fonction d'encodage sinusoïdale pour transformer le pas de temps $t$ en un vecteur d'embedding de dimension 512, qui est ensuite projeté à la dimension des canaux du modèle de diffusion et injecté dans les blocks de ResNet.\\


\subsubsection{Encodage de la Classe}

L'encodage de la classe (ou de la condition $y$ plus généralement), quant à lui, est réalisé suivant une table de correspondance, qui associe à chaque classe un vecteur d'embedding de dimension 512. Par exemple, pour le jeu de données MNIST, qui comporte 10 classes (les chiffres de 0 à 9), nous avons une table d'embeddings de taille $10 \times 512$. Lors de l'entraînement, pour les exemples conditionnels, nous récupérons l'embedding correspondant à la classe $y$ associée à l'image, tandis que pour les exemples inconditionnels, nous utilisons un token spécial (un vecteur nul) pour représenter l'absence de conditionnement.\\

La figure \ref{fig:class_encoding} illustre l'encodage de la classe.\\

\begin{figure}[htpb]
\centering
    \resizebox{0.8\textwidth}{!}{%
        \begin{tikzpicture}[node distance=1cm, every node/.style={draw, font=\scriptsize, align=center}]
            \node[fill=gray!10] (y) {$y$ (classe)};
            \node[fill=blue!5, right=of y] (emb) {Table d'embeddings\\nn.Embedding (10, 512)};
            \node[fill=red!5, right=of emb, aspect=1.5] (drop) {Dropout de classe\\P(dropout)=0.2};
            \node[rounded corners, fill=green!10, right=of drop] (out) {Vecteur $y_{emb}$ (512)};
            
            \draw[-{Stealth}, thick] (y) -- (emb); 
            \draw[-{Stealth}, thick] (emb) -- (drop); 
            \draw[-{Stealth}, thick] (drop) -- (out);
        \end{tikzpicture}%
    }
    \caption{Schéma de l'encodage de la classe dans un DDPM avec Classifier-Free Guidance.}
    \label{fig:class_encoding}
\end{figure}

Nous avons alors un vecteur d'embedding de classe $y_{emb}$ de dimension 512, qui est injecté dans les différentes "couches" du modèle de diffusion, de la même manière que le vecteur d'encodage du pas de temps.\\

Notons que dans notre implémentation, nous avons choisi d'appliquer un dropout de classe avec une probabilité de 0.2, ce qui signifie que 20\% des exemples présentés au modèle pendant l'entraînement sont inconditionnels (avec un token nul pour la classe), tandis que les 80\% restants sont conditionnels (avec la classe associée). Avec un jeu de données de taille suffisante, ce ratio permet au modèle d'apprendre à générer des images réalistes à la fois avec et sans conditionnement, ce qui est essentiel pour le bon fonctionnement de la Classifier-Free Guidance.\\

Précisons aussi que les paramètres de la table de correspondance des classes (les embeddings) sont appris conjointement avec les autres paramètres du modèle de diffusion pendant l'entraînement, ce qui permet au modèle d'apprendre des représentations de classe adaptées à la tâche de génération.\\

\subsubsection{Architecture d'un block de ResNet conditionnel}

Le fonctionnement du block de ResNet conditionnel est similaire à celui d'un block de ResNet inconditionnel. L'ajout de mécanismes pour intégrer la classe (équation \ref{eq:resnet_modulation}) permet au modèle de moduler les canaux de manière conditionnelle, de manière à extraire des caractéristiques spécifiques à la classe et au niveau de bruit présent dans l'image à chaque étape du processus de diffusion.\\

\begin{equation}
\text{out} = \text{out} \times t_{prj} + y_{prj}
\label{eq:resnet_modulation}
\end{equation}

avec:
\begin{itemize}
    \item $\text{out}$ est la sortie de la première convolution du block de ResNet, avant la modulation.
    \item $t_{prj}$ est le vecteur obtenu à partir de l'encodage du pas de temps, projeté à la dimension des canaux du block.
    \item $y_{prj}$ est le vecteur obtenu à partir de l'encodage de la classe, projeté à la dimension des canaux du block.
\end{itemize} 

Nous avons alors une modulation multiplicative (scale) par le vecteur du pas de temps, qui permet au modèle d'ajuster l'intensité des caractéristiques extraites en fonction du niveau de bruit présent dans l'image, et une modulation additive (shift) par le vecteur de la classe, qui permet au modèle d'ajuster les caractéristiques extraites, en fonction de la classe à laquelle l'image doit appartenir (nous pourrions imaginer que le conditionnement de classe "pousse" vers une région spécifique de l'espace des images).\\

L'architecture complète d'un block de ResNet conditionnel pour un DDPM avec Classifier-Free Guidance est présentée en annexe \ref{annexe:cfg_resnet_architecture}.\\

\subsection{Algorithmes : Entraînement et Inférence}

Ayant décrit la prise en compte du pas de temps et de la classe dans l'architecture du modèle de diffusion conditionnel, nous pouvons maintenant détailler les algorithmes d'entraînement et d'inférence pour un DDPM avec Classifier-Free Guidance.\\

\subsubsection{Algorithme d'entraînement pour un DDPM avec Classifier-Free Guidance}

Le principe de l'entraînement d'un DDPM avec Classifier-Free Guidance est similaire à celui d'un DDPM inconditionnel, avec l'ajout de la construction du conditionnement de classe (ou de la condition $y$ plus généralement) et de la présentation d'exemples à la fois conditionnels et inconditionnels au modèle pendant l'entraînement.\\

Le principe de l'entraînement est le même que pour un DDPM inconditionnel, à savoir que nous présentons au modèle des images bruitées $x_t$ à différents pas de temps $t$, et que nous lui demandons de prédire le bruit $\epsilon$ ajouté à l'image. La principale différence réside dans la construction du conditionnement de classe, où nous construisons, pour chaque image, l'embedding de classe $y_{emb}$ à partir de la table d'embeddings, en utilisant la classe $y$ associée, et en appliquant un dropout de classe pour obtenir à la fois des exemples conditionnels et inconditionnels.
Alors, nous pouvons prédire le bruit $\epsilon$ à partir de l'image bruitée $x_t$, du pas de temps $t$, et de l'embedding de classe $y_{emb}$, en utilisant le modèle de diffusion conditionnel $\epsilon_\theta(x_t, t, y_{emb})$. Nous calculons ensuite la perte associée, et mettons à jour les paramètres du modèle par une descente de gradient.\\

L'algorithme d'entraînement complet pour un DDPM avec Classifier-Free Guidance est présenté en annexe \ref{annexe:algos}.\\

\subsubsection{Algorithme d'inférence pour un DDPM avec Classifier-Free Guidance}

Le principe de l'inférence pour un DDPM avec Classifier-Free Guidance est également similaire à celui d'un DDPM inconditionnel, avec l'ajout de la combinaison des scores conditionnel et inconditionnel à chaque étape du processus de génération, afin de guider la génération vers la condition souhaitée.\\

À chaque pas de temps $t$, nous prédisons à la fois le bruit conditionnel $\epsilon_t^{cond} = \epsilon_\theta(x_t, t, y_{emb})$ et le bruit inconditionnel $\epsilon_t^{uncond} = \epsilon_\theta(x_t, t, \text{token nul})$. Nous combinons ensuite ces deux prédictions pour obtenir une prédiction guidée $\hat{\epsilon}_t$ selon la formule \ref{eq:cfg_combined_s} :\\

\begin{equation}
\hat{\epsilon}_t = (1 + s) \epsilon_t^{cond} - s \epsilon_t^{uncond}
\label{eq:cfg_combined_s}
\end{equation}

où $s$ est le facteur d'échelle de guidage, qui nous permet de contrôler l'intensité du guidage vers la condition $y$. \\

Nous utilisons ensuite cette prédiction guidée $\hat{\epsilon}_t$ pour calculer la mise à jour de $x_t$ et obtenir $x_{t-1}$, en suivant les étapes habituelles d'un DDPM (calcul de l'image $\mu_t$ à partir de $x_t$ et $\hat{\epsilon}_t$, ajout de bruit, etc.).\\

L'algorithme d'inférence complet pour un DDPM avec Classifier-Free Guidance est présenté en annexe \ref{annexe:algos}.\\

\underline{Note:} La formule utilisée dans l'algorithme d'inférence pour combiner les scores conditionnel et inconditionnel est une reformulation de la formule \ref{eq:cfg_combined}, où nous avons posé $s \leftarrow 1 + s$, et appliqué la formule dans le cadre d'un DDPM. Pour $s=0$, nous avons une génération purement conditionnelle, pour $s>0$, nous avons un guidage renforcé, et pour $s<0$, nous avons un guidage atténuée (inconditionnelle pour $s=-1$).\\

\section{Résultats expérimentaux}

Les résultats expérimentaux présentés dans cette section ont été obtenus en appliquant les algorithmes d'entraînement et d'inférence détaillés précédemment à un modèle de diffusion avec Classifier-Free Guidance, suivant les configurations d'architecture définies précédemment, sur le jeu de données MNIST. Le modèle a été entraîné pendant 120 epochs, avec un batch size de 128, et une probabilité de dropout de classe de 0.2.\\

Afin de permettre une observation plus aisée des détails de génération, des versions agrandies des planches de résultats sont consultables en annexe \ref{annexe:figures_grandes}.\\

L'implémentation a été réalisée en PyTorch, et se base sur une implémentation de \textit{T. Matsuzaki} \cite{tsmatz_cfg}, que nous avons adaptée et modifiée pour notre projet.\\

\subsection{Qualité de la génération inconditionnelle/conditionnelle sur MNIST}

Nous avons évalué la qualité de la génération à la fois en mode inconditionnel (sans conditionnement de classe, $s=-1$) et en mode conditionnel (avec conditionnement de classe, $s=0$), en générant des échantillons à partir du modèle entraîné. \\

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.25\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/guided_unet_5___five_s-1.0.png}
        \caption{Classe $\emptyset$ | $s=-1$}
        \label{fig:cfg_uncond}
    \end{subfigure}%
    \hfill
    \begin{subfigure}[b]{0.25\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/guided_unet_5___five_s-0.5.png}
        \caption{Classe 5 | $s=-0.5$}
        \label{fig:cfg_uncond_cond}
    \end{subfigure}%
    \hfill
    \begin{subfigure}[b]{0.25\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/guided_unet_5___five_s0.0.png}
        \caption{Classe 5 | $s=0$}
        \label{fig:cfg_cond0}
    \end{subfigure}%
    \hfill
    \begin{subfigure}[b]{0.25\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/guided_unet_5___five_s10.0.png}
        \caption{Classe 5 | $s=10$}
        \label{fig:cfg_cond10}
    \end{subfigure}

    \caption{Exemples d'images générées par un DDPM avec Classifier-Free Guidance (classe 5).}
    \label{fig:cfg_results}
\end{figure}

La figure \ref{fig:cfg_results} montre que le modèle est capable de générer des images réalistes à la fois en mode inconditionnel (figure \ref{fig:cfg_uncond}) et en mode conditionnel (figure \ref{fig:cfg_cond0}, \ref{fig:cfg_cond10}). En mode inconditionnel, les images générées sont variées et ne correspondent pas à une classe spécifique, tandis qu'en mode conditionnel, les images générées sont clairement reconnaissables comme appartenant à la classe 5. En augmentant encore davantage le facteur d'échelle de guidage (figure \ref{fig:cfg_cond10}, $s=10$), nous observons que les images générées sont encore plus fortement guidées vers la classe 5, mais au prix d'une perte de diversité: les images générées sont très similaires les unes aux autres, et peuvent presque sembler saturées par les caractéristiques de la classe 5 (les traits sont très épais et marqués afin de maximiser la reconnaissance de la classe 5).\\

En revanche, en mode "semi"-conditionnel ($s=-0.5$), nous observons un semblant d'entre-deux, où les images générées présentent des caractéristiques de la classe 5, mais ne sont pas aussi nettes et reconnaissables que dans le mode conditionnel pur ($s=0$). Les images générées sont dans un espace intermédiaire et qui se trouve à priori ni dans la région de l'espace des images correspondant à la classe 5, ni dans la région correspondant à une génération inconditionnelle. Nous avons donc une génération qui est à la fois influencée par la classe cible, mais aussi par les caractéristiques générales des images du jeu de données, mais où les images perdent en qualité et en réalisme.\\

\underline{Note:} Nous avons également entrainé un modèle de diffusion conditionnel avec Classifier-Free Guidance sur le jeu de données CIFAR-10. L'entraînement a été réalisé pendant 300 epochs, avec un batch size de 128, et une probabilité de dropout de classe de 0.2. Les résultats obtenus sont donnés en annexe \ref{annexe:cfg_cifar10}, et présentent certaines limites du modèle. 

\section{Conclusion}

Ce projet nous a permis d'étudier et d'implémenter de manière progressive les différents mécanismes de guidage pour la génération d'images par les modèles de diffusion.\\

Nous avons d'abord abordé les modèles DDPM dans leur formulation de base. Si ces derniers démontrent une excellente capacité à produire des échantillons fidèles à la distribution de données initiale et suffisamment diversifiés, ils présentent une limite majeure : nous n'avons à priori pas de contrôle sur les caractéristiques des images générées, ce qui peut être problématique dans de nombreuses applications où nous souhaitons générer des images avec des propriétés spécifiques (par exemple, appartenant à une classe donnée).\\

La \textit{Classifier Guidance} a donc été introduite pour pallier à cette limitation, en exploitant un classifieur externe pour guider le processus de génération vers une classe cible. Cette approche permet d'obtenir un contrôle explicite sur les caractéristiques des images générées, mais elle présente également des inconvénients importants. En effet, si cette méthode a montré des résultats relativement satisfaisants en termes de qualité et de respect de la conditionnalité, elle reste coûteuse en pratique. En effet, elle nécessite d'entraîner et d'évaluer à chaque pas de temps un classifieur spécifiquement robuste au bruit, ce qui alourdit considérablement l'architecture globale ainsi que la phase d'inférence.\\

La \textit{Classifier-Free Guidance} apporte alors une solution à la fois plus simple et plus performante. En adaptant l'entraînement de notre modèle pour apprendre conjointement les distributions conditionnelle et inconditionnelle, nous parvenons à guider la génération de manière interne, sans recourir à un modèle tiers. 
Les résultats expérimentaux obtenus confirment que cette méthode permet de parvenir à un compromis entre qualité, diversité et respect de la conditionnalité, par le simple ajustement du facteur d'échelle $s$.\\

À ce titre, le standard de la génération conditionnelle par les modèles de diffusion est désormais la \textit{Classifier-Free Guidance}, qui est largement utilisée dans les modèles de diffusion les plus récents (Stable Diffusion, DALL-E 2, etc.), et qui a permis d'obtenir des résultats plus que convaincants en termes de qualité et de respect de la conditionnalité.\\

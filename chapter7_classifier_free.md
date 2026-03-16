\chapter{Classifier-Free Guidance}
\label{chap:classifier_free}
\rhead{Classifier-Free Guidance}

\section{Motivations}

Dans le chapitre précédent, nous avons détaillé comment la Classifier Guidance permet de contrôler le processus de génération des modèles de diffusion en utilisant le gradient issu d'un modèle de classification. Bien que cette méthode soit efficace, elle s'accompagne de contraintes majeures qui limitent son application pratique.\\

La principale difficulté réside dans la nécessité de concevoir, d'entraîner et d'intégrer un réseau de neurones externe  supplémentaire. Ce classifieur doit être suffisamment robuste pour classifier des images soumises à des niveaux de bruit qui peuvent être extrêmes (pour des valeurs des pas de temps $t$ proches de $T$), afin de correspondre aux états intermédiaires $x_t$ du processus de diffusion.\\

Pour pallier à ces limitations, à la fois architecturales et pratiques, Jonathan Ho et Tim Salimans ont introduit en 2021 la \textbf{Classifier-Free Guidance} \cite{ho2022}. L'objectif de cette approche est de permettre un conditionnement du processus de génération sans pour autant nécessiter de modèle de classification externe.\\

\section{Principe général de la Classifier-Free Guidance}

Contrairement à la Classifier Guidance, qui s'appuie sur un modèle de classification pour guider la génération, la Classifier-Free Guidance repose sur l'idée d'entraîner un modèle de diffusion conditionnel qui prend en entrée non seulement une image bruitée $x_t$ et le pas de temps $t$, mais aussi une condition $y$. L'objectif est d'entraîner le modèle sur des exemples conditionnels (où la condition $y$ associée est fournie), mais aussi sur des exemples inconditionnels (où la condition est remplacée par un token nul), afin de permettre au modèle d'apprendre à générer des images à la fois avec et sans conditionnement.\\

Dans cette section, nous présentons les fondements de la Classifier-Free Guidance, en détaillant dans un premier temps sa formulation mathématique, puis en proposant une intuition géométrique pour mieux comprendre son fonctionnement (basé sur les travaux S. Dieleman \cite{dieleman2023}).\\

\underline{Note:} Dans la suite, nous considérerons que $y$ représente une classe, mais la méthode est applicable à d'autres types de conditions (texte, attributs, etc.).\\

\subsection{Formulation mathématique de la CFG (Classifier-Free Guidance)}

L'idée fondamentale de la Classifier-Free Guidance (CFG) repose sur une reformulation algébrique du théorème de Bayes appliqué aux fonctions de score.\\

Si nous reprenons la formulation de la Classifier Guidance, nous avons :\\

\begin{equation}
\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(x_t) + \gamma \nabla_{x_t} \log p(y | x_t)
\label{eq:cg_initial}
\end{equation}

\vspace{0.3cm}

où $\nabla_{x_t} \log p(y | x_t)$ est le gradient de la log-vraisemblance de la classe $y$, étant donnée l'image bruitée $x_t$, qui est estimé à l'aide d'un classifieur externe, et $\gamma$ est un hyperparamètre de guidance qui contrôle l'intensité du conditionnement.\\

D'après le \textbf{théorème de Bayes}, nous avons:\\

\begin{equation}
\nabla_{x_t} \log p(y | x_t) = \nabla_{x_t} \log p(x_t | y) - \nabla_{x_t} \log p(x_t)
\label{eq:cfg_bayes}
\end{equation}

\vspace{0.3cm}

Cette formule est cruciale, dans la mesure où elle nous permet d'obtenir, simplement en soustrayant le score inconditionnel du modèle conditionnel, le score d'un classifieur externe. \\

Nous pouvons réécrire l'expression initiale de la manière suivante : \\

\begin{equation}
\nabla_{x_t} \log p(x_t | y) = (1 - \gamma) \nabla_{x_t} \log p(x_t) + \gamma \nabla_{x_t} \log p(x_t | y)
\label{eq:cfg_combined}
\end{equation}

\vspace{0.3cm}

Nous avons alors une combinaison linéaire entre le score inconditionnel $\nabla_{x_t} \log p(x_t)$ et le score conditionnel $\nabla_{x_t} \log p(x_t | y)$, pondérée par le paramètre $\gamma$, qui nous permet, sans classifieur externe, de conditionner le processus de génération. \\

Lors de l'entraînement du modèle, nous présentons à la fois des exemples conditionnels (où la classe $y$ est donnée) et des exemples inconditionnels (où la classe est remplacée par un token nul, on parle de "dropout" de classe). Le modèle de diffusion est alors capable de générer des images à la fois avec et sans conditionnement. Lors de l'inférence, à chaque pas de temps, nous pouvons alors calculer les deux scores (conditionnel et inconditionnel) et les combiner pour guider la génération selon le niveau de guidance souhaité. \\

Selon les valeurs de $\gamma$, nous pouvons obtenir différents comportements de génération :
\begin{itemize}
    \item $\gamma = 0$ : Génération purement inconditionnelle, où le modèle génère des images sans tenir compte de la condition $y$.
    \item $\gamma = 1$ : Génération purement conditionnelle, où le modèle génère des images strictement conformes à la condition $y$.
    \item $\gamma > 1$ : Génération avec une guidance renforcée, où le modèle est fortement incité à suivre la condition $y$, au risque de réduire la diversité des échantillons générés. \\
\end{itemize}

Nous pouvons maintenant présenter une intuition géométrique de la CFG, qui devrait illustrer de manière plus visuelle comment la combinaison des scores conditionnel et inconditionnel permet de guider le processus de génération dans l'espace des images. \\

\subsection{Intuition géométrique de la CFG}

Considérons une image bruitée $x_t$ à un pas de temps donné $t$. Dans l'espace des images, nous pouvons imaginer que les différentes classes forment des régions distinctes. \\

À l'instant $t$, le modèle de diffusion effectue deux prédictions: celle de $\hat{x}_0$ inconditionnelle (sans conditionnement) et celle de $\hat{x}_0 | y$ conditionnelle. Ces deux prédictions peuvent être représentées comme des points dans l'espace des images. La figure \ref{fig:cfg_predictions} illustre cette situation, où nous avons une prédiction inconditionnelle et une prédiction conditionnelle. \\

La formule \ref{eq:cfg_bayes} nous indique que le score de la CFG est une combinaison linéaire entre le score inconditionnel et le score conditionnel. Géométriquement, cela signifie que nous pouvons représenter ce score comme un vecteur "delta" qui va de la prédiction inconditionnelle vers la prédiction conditionnelle, comme illustré dans la figure \ref{fig:cfg_delta}. \\

En multipliant ce vecteur "delta" par un facteur de guidance $\gamma$, nous pouvons ajuster l'intensité de ce vecteur, ce qui correspond à renforcer ou atténuer la guidance vers la condition $y$. La figure \ref{fig:cfg_scale_delta} montre comment le vecteur "delta" est mis à l'échelle par le paramètre $\gamma$. \\

Ensuite, lors de la mise à jour de $x_t$ pour obtenir $x_{t-1}$, nous ajoutons ce vecteur de score à la prédiction inconditionnelle, ce qui nous permet de guider la génération vers la condition souhaitée. La figure \ref{fig:cfg_step} illustre cette étape, où le vecteur de score est utilisé pour ajuster la prédiction inconditionnelle. \\

Enfin, l'ajout de bruit pour obtenir $x_{t-1}$ à partir de la prédiction ajustée est représenté dans la figure \ref{fig:cfg_add_noise}. Cette étape est très importante dans la génération. En effet, si elle n'avait pas lieu, la génération deviendrait alors déterministe, et nous perdrions la diversité des échantillons générés. L'ajout de bruit permet de maintenir une certaine variabilité dans les échantillons, même lorsque la guidance est forte. Notons que nous n'ajoutons pas de bruit lors de la dernière étape de génération (lorsque $t=1$), afin d'obtenir une image finale nette. \\

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
        \caption{Vecteur "delta" entre les prédictions}
        \label{fig:cfg_delta}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/cfg_geometry_scale_delta.png}
        \caption{Vecteur "delta" mis à l'échelle par $\gamma$}
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

Nous avons ainsi une intuition géométrique de la CFG, qui nous permet de visualiser comment les différentes composantes du score interagissent pour guider le processus de génération dans l'espace des images. Cette perspective peut être très utile pour comprendre les effets du paramètre de guidance $\gamma$ et pour interpréter les résultats expérimentaux que nous présenterons par la suite. \\

\underline{Note:} Les figures \ref{fig:cfg_predictions}, \ref{fig:cfg_delta}, \ref{fig:cfg_scale_delta}, et \ref{fig:cfg_add_noise} montrent des espaces en 2 dimensions pour illustrer les différentes composantes du score et leur combinaison. Il est important de noter que ces figures sont ici pour aider à la compréhension, mais que l'espace réel des images est de beaucoup plus haute dimension.

\section{Application aux DDPMs}

Ayant présenté et donné une intuition géométrique de la Classifier-Free Guidance, nous pouvons maintenant appliquer cette méthode aux modèles de diffusion basés sur les DDPMs. 

\subsection{Intégration des Pas de Temps et des Classes}



\subsection{Algorithmes : Entraînement et Inférence}


\section{Résultats expérimentaux}

\subsection{Qualité de la génération inconditionnelle/conditionnelle sur MNIST}
\subsection{Impact de l'échelle de \textit{Guidance}}






\chapter{Classifier-Free Guidance}
\label{chap:classifier}
\rhead{Classifier-Free Guidance}

\section{Motivations}
\label{sec:cfg_motivations}

% Transition avec le chapitre 6 : rappeler les limites de la Classifier Guidance 
% (nécessité d'entraîner un classifieur distinct résistant au bruit, complexité du pipeline).
% Introduction des motivations pour une méthode sans classifieur externe.

Dans le chapitre précédent, nous avons détaillé comment la Classifier Guidance permet de contrôler le processus de génération des modèles de diffusion en utilisant le gradient issu d'un modèle de classification. Bien que cette méthode soit efficace, elle s'accompagne de contraintes majeures qui limitent son application pratique. 


Pour palier à ces limitations, à la fois architecturales et pratiques, Jonathan Ho et Tim Salimans ont introduit en 2021 la \textbf{Classifier-Free Guidance} \cite{ho2021classifierfree}. L'objectif de cette approche est de permettre un conditionnement du processus de génération sans pour autant nécessiter de modèle de classification externe.

\section{Principe général de la Classifier-Free Guidance}
\label{sec:cfg_general}

Contrairement à la Classifier Guidance, qui s'appuie sur un modèle de classification pour guider la génération, la Classifier-Free Guidance repose sur l'idée d'entraîner un modèle de diffusion conditionnel qui prend en entrée non seulement une image bruitée $x_t$ et le pas de temps $t$, mais aussi une condition $y$. L'objectif est d'entraîner le modèle sur des exemples conditionnels (où la condition $y$ associée est fournie), mais aussi sur des exemples inconditionnels (où la condition est remplacée par un token nul), afin de permettre au modèle d'apprendre à générer des images à la fois avec et sans conditionnement.

Dans cette section, nous présentons les fondements de la Classifier-Free Guidance, en détaillant dans un premier temps sa formulation mathématique, puis en proposant une intuition géométrique pour mieux comprendre son fonctionnement (basé sur les travaux S. Dieleman \cite{dieleman2023}).

\underline{Note:} Dans la suite, nous considérerons que $y$ représente une classe, mais la méthode est applicable à d'autres types de conditions (texte, attributs, etc.).

\subsection{Formulation mathématique de la CFG (Classifier-Free Guidance)}

L'idée fondamentale de la Classifier-Free Guidance (CFG) repose sur une reformulation algébrique du théorème de Bayes appliqué aux fonctions de score. 

Si nous reprenons la formulation de la Classifier Guidance, nous avons :

$$
\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(x_t) + \omega \nabla_{x_t} \log p(y | x_t)
$$

où $\nabla_{x_t} \log p(y | x_t)$ est le gradient de la log-vraisemblance de la classe $y$, étant donnée l'image bruitée $x_t$, qui est estimé à l'aide d'un classifieur externe, et $\omega$ est un hyperparamètre de guidance qui contrôle l'intensité du conditionnement.

D'après le \textbf{théorème de Bayes}, nous avons:

$$
\nabla_{x_t} \log p(y | x_t) = \nabla_{x_t} \log p(x_t | y) - \nabla_{x_t} \log p(x_t)
$$

Cette formule est cruciale, dans la mesure où elle nous permet d'obtenir, simplement en soustrayant le score inconditionnel du modèle conditionnel, le score d'un classifieur externe.

Nous pouvons réécrire l'expression initiale de la manière suivante :

$$
\nabla_{x_t} \log p(x_t | y) = (1 - \omega) \nabla_{x_t} \log p(x_t) + \omega \nabla_{x_t} \log p(x_t | y)
$$

Nous avons alors une combinaison linéaire entre le score inconditionnel $\nabla_{x_t} \log p(x_t)$ et le score conditionnel $\nabla_{x_t} \log p(x_t | y)$, pondérée par le paramètre $\omega$, qui nous permet, sans classifieur externe, de conditionner le processus de génération.

Lors de l'entraînement du modèle, nous présentons à la fois des exemples conditionnels (où la classe $y$ est donnée) et des exemples inconditionnels (où la classe est remplacée par un token nul, on parle de "dropout" de classe). Le modèle de diffusion est alors capable de générer des images à la fois avec et sans conditionnement. Lors de l'inférence, à chaque pas de temps, nous pouvons alors calculer les deux scores (conditionnel et inconditionnel) et les combiner pour guider la génération selon le niveau de guidance souhaité.

Selon les valeurs de $\omega$, nous pouvons obtenir différents comportements de génération :
- $\omega = 0$ : Génération purement inconditionnelle, où le modèle génère des images sans tenir compte de la condition $y$.
- $\omega = 1$ : Génération purement conditionnelle, où le modèle génère des images strictement conformes à la condition $y$.
- $\omega > 1$ : Génération avec une guidance renforcée, où le modèle est fortement incité à suivre la condition $y$, au risque de réduire la diversité des échantillons générés.

Nous pouvons maintenant présenter une intuition géométrique de la CFG, qui devrait illustrer de manière plus visuelle comment la combinaison des scores conditionnel et inconditionnel permet de guider le processus de génération dans l'espace des images.

\subsection{Intuition géométrique de la CFG}

Considérons une image bruitée $x_t$ à un pas de temps donné $t$. Dans l'espace des images, nous pouvons imaginer que les différentes classes forment des régions distinctes.

À l'instant



\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.24\textwidth}
        \centering
        \includegraphics[width=\textwidth]{report/figures/cfg_geometry_predict_cond.png}
        \caption{Prédiction inconditionnelle et conditionnelle}
        \label{fig:cfg_predictions}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.24\textwidth}
        \centering
        \includegraphics[width=\textwidth]{report/figures/cfg_geometry_delta.png}
        \caption{Vecteur "delta" entre les prédictions}
        \label{fig:cfg_delta}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.24\textwidth}
        \centering
        \includegraphics[width=\textwidth]{report/figures/cfg_geometry_scale_delta.png}
        \caption{}
        \label{fig:cfg_scale_delta}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.24\textwidth}
        \centering
        \includegraphics[width=\textwidth]{report/figures/cfg_geometry_add_noise.png}
        \caption{Guidance CFG}
        \label{fig:cfg_add_noise}
    \end{subfigure}
    \caption{Représentation géométrique simplifiée de la Classifier-Free Guidance}
    \label{fig:cfg_geometry}
\end{figure}

\underline{Note:} Les figures \ref{}, \ref{}, \ref{}, et \ref{} montrent des espaces en 2 dimensions pour illustrer les différentes composantes du score et leur combinaison. Il est important de noter que ces figures sont ici pour aider à la compréhension, mais que l'espace réel des images est de beaucoup plus haute dimension.

\section{Application aux DDPMs}
\label{sec:cfg_ddpms}
% Focus exclusif sur l'application de la CFG au sein du formalisme des Denoising Diffusion Probabilistic Models.

\subsection{Architecture du modèle conditionnel}
\label{subsec:cfg_architecture}
% Présentation de l'architecture typique (souvent un U-Net adapté pour le conditionnement).

\subsection{Intégration des Pas de Temps et des Classes}
\label{subsec:cfg_embeddings}
% Explication de la création et de l'injection des embeddings :
% - Temps : Sinusoidal positional embeddings interpolés.
% - Classe : Vecteurs appris (nn.Embedding).
% - Comment ces embeddings sont combinés et injectés dans les blocs résiduels de l'architecture.
% - Le mécanisme de "dropout" de classe (remplacement aléatoire de la classe par un token nul) pendant l'entraînement.

\subsection{Algorithmes : Entraînement et Inférence}
\label{subsec:cfg_algorithms}
% Formalisation des boucles d'entraînement et du processus d'échantillonnage spécifique à la CFG avec DDPM.

\section{Résultats expérimentaux}
\label{sec:cfg_results}
% Analyse des résultats de la génération conditionnelle sur la base de données MNIST.

\subsection{Qualité de la génération inconditionnelle/conditionnelle sur MNIST}
\label{subsec:cfg_mnist_quality}
% Présentation visuelle des générations par classe.

\subsection{Impact de l'échelle de \textit{Guidance} ($w$)}
\label{subsec:cfg_guidance_scale}
% Examen de l'effet de l'augmentation ou de la diminution du paramètre w sur la diversité et la fidélité de l'image.

% Remarque : Mentionner ici ou dans l'introduction du chapitre que des essais similaires sur CIFAR10 
% sont disponibles en Annexe X pour démontrer la scalabilité de l'approche.

\section{Conclusion}
\label{sec:cfg_conclusion}
% Résumé des apports de la CFG par rapport au Chapitre 6 et ouverture vers les chapitres suivants.



\appendix

\section{Reformulation algébrique de la Classifier Guidance}

$$
\begin{aligned}
\nabla_{x_t} \log p(x_t | y) &= \nabla_{x_t} \log p(x_t) + \omega \nabla_{x_t} \log p(y | x_t) \\
&= \nabla_{x_t} \log p(x_t) + \omega \nabla_{x_t} \log \left( \frac{p(x_t | y) p(y)}{p(x_t)} \right) \\
&= \nabla_{x_t} \log p(x_t) + \omega \nabla_{x_t} \log p(x_t | y) + \omega \nabla_{x_t} \log p(y) - \omega \nabla_{x_t} \log p(x_t) \\
&= (1 - \omega) \nabla_{x_t} \log p(x_t) + \omega \nabla_{x_t} \log p(x_t | y)
\end{aligned}
$$
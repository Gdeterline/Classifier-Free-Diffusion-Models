\chapter{Modèles de Diffusion}
\label{chap:intro_modeles}
\rhead{Modèles de Diffusion}

\section{Modèles de Diffusion}

Les modèles de diffusion constituent une nouvelle classe de modèles génératifs à l'état de l'art, capables de produire des images de haute résolution. L'idée principale de ces modèles repose sur la correction itérative de la prédiction à chaque pas de temps, jusqu'à l'obtention d'une génération finale cohérente. 

Dans cette partie, nous allons aborder les principes de base ainsi que l'architecture de ces modèles. Pour l'élaboration de ce chapitre, nous nous concentrerons particulièrement sur les modèles probabilistes de diffusion par débruitage (\textit{Denoising Diffusion Probabilistic Models}, DDPM) proposés par \cite{ho2022}, tout en notant qu'il existe d'autres approches basées sur la fonction de score (\textit{score-based models}).
\subsection{Processus de diffusion directe (Forward Process)}

Les modèles de diffusion sont formulés de manière markovienne, c'est-à-dire comme une chaîne de Markov de $T$ étapes. Cette propriété implique que chaque étape dépend uniquement de l'état atteint à l'étape précédente.

Supposons que nous disposions d'un point de données $x_0$ échantillonné à partir de la distribution réelle $q(x)$ ($x_0 \sim q(x)$). Nous pouvons définir un processus de diffusion directe en ajoutant progressivement du bruit à chaque étape de la chaîne. Plus précisément, à chaque itération $t$, on injecte un bruit gaussien de variance $\beta_t$ à l'échantillon $x_{t-1}$, produisant ainsi une nouvelle variable latente $x_t$. La distribution de transition $q(x_t | x_{t-1})$ de ce processus peut être formulée comme suit :

\begin{equation}
q(x_t | x_{t-1}) = \mathcal{N}(x_t ; \mu_t = \sqrt{1 - \beta_t} x_{t-1}, \Sigma_t = \beta_t \mathbf{I})
\end{equation}

Où $\beta_t$ représente le schéma de bruit (\textit{variance schedule}) à l'étape $t$ et $\mathbf{I}$ est la matrice identité.
\subsection{L'astuce de reparamétrage }

Puisque nous évoluons dans un scénario multidimensionnel, $\mathbf{I}$ représente la matrice identité, indiquant que chaque dimension possède la même variance $\beta_t$. Il est important de noter que $q(x_t | x_{t-1})$ suit toujours une distribution normale, définie par une moyenne $\mu_t = \sqrt{1 - \beta_t} x_{t-1}$ et une variance $\Sigma_t = \beta_t \mathbf{I}$. Ici, $\Sigma_t$ est systématiquement une matrice diagonale de variances.

Dès lors, nous pouvons passer de la donnée initiale $x_0$ à l'état bruité $x_T$ sous une forme fermée (\textit{closed form}) de manière traitable. Mathématiquement, cela correspond à la probabilité \textit{a posteriori} définie par la trajectoire :
\begin{equation}
q(x_{1:T} | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})
\end{equation}

L'utilisation du symbole "$:$" dans $q(x_{1:T})$ indique l'application répétée de $q$ du pas de temps $1$ jusqu'à $T$. Cependant, pour un pas de temps intermédiaire (par exemple $t=500$), il serait inefficace d'appliquer $q$ 500 fois pour échantillonner $x_t$. 

L'astuce de reparamétrage (\textit{reparametrization trick}) offre une solution directe. En définissant $\alpha_t = 1 - \beta_t$ et $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$, nous pouvons prouver par récurrence que :
\begin{equation}
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \text{avec } \epsilon \sim \mathcal{N}(0, \mathbf{I})
\end{equation}

Ainsi, pour produire un échantillon $x_t$, nous utilisons la distribution suivante :
\begin{equation}
q(x_t | x_0) = \mathcal{N}(x_t ; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})
\end{equation}

Comme le schéma de bruit $\beta_t$ est un hyperparamètre, nous pouvons précalculer $\alpha_t$ et $\bar{\alpha}_t$ pour tous les pas de temps. Cela permet d'échantillonner $x_t$ à n'importe quel instant $t$ de manière arbitraire et instantanée, ce qui facilitera le calcul de notre fonction de perte $L_t$.
\subsection{Schéma de variance (\textit{Variance Schedule})}

Le paramètre de variance $\beta_t$ peut être soit fixé comme une constante, soit choisi selon un calendrier spécifique (\textit{schedule}) évoluant au cours du temps $T$. En pratique, on peut définir un schéma de variance de type linéaire, quadratique, ou encore cosinus. 

Les auteurs originaux de DDPM \cite{ho2022} ont utilisé un calendrier linéaire augmentant de $\beta_1 = 10^{-4}$ à $\beta_T = 0.02$. Cependant, \cite{nichol2022} a montré que l'emploi d'un $\beta$ basé sur une fonction cosinus (\textit{cosine schedule}) offre de meilleures performances. Ce dernier permet notamment de ralentir l'ajout de bruit aux étapes proches de $x_0$, préservant ainsi davantage la structure de l'image durant les phases critiques du processus.
\begin{figure}[h!]
    \centering
    \includegraphics[width=\linewidth]{images/variance-schedule.png}
\caption{Échantillons latents issus respectivement des $\beta$ linéaire (en haut) et cosinus (en bas).\cite{dhariwal2021}}
\end{figure}
\subsection{Processus de diffusion inverse (\textit{Reverse Diffusion})}

Lorsque $T \to \infty$, l'échantillon latent $x_T$ devient quasiment une distribution gaussienne isotrope. Par conséquent, si nous parvenons à apprendre la distribution inverse $q(x_{t-1} | x_t)$, nous pouvons échantillonner $x_T$ à partir de $\mathcal{N}(0, \mathbf{I})$, exécuter le processus inverse et obtenir un échantillon de $q(x_0)$. Cela permet de générer un nouveau point de donnée issu de la distribution d'origine. La question centrale est alors de savoir comment modéliser ce processus de diffusion inverse.

\subsubsection{Approximation du processus inverse par un réseau de neurones}

En pratique, la distribution $q(x_{t-1} | x_t)$ est intraitable car son estimation statistique nécessiterait des calculs impliquant l'ensemble de la distribution de données. À la place, nous l'approximons par un modèle paramétré $p_\theta$ (par exemple, un réseau de neurones). 

Puisque $q(x_{t-1} | x_t)$ est également gaussienne pour des valeurs de $\beta_t$ suffisamment petites, nous pouvons choisir $p_\theta$ comme une distribution gaussienne et paramétrer simplement sa moyenne et sa variance :

\begin{equation}
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1} ; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
\end{equation}

Si nous appliquons cette formule inverse pour tous les pas de temps ($p_\theta(x_{0:T})$, également appelée trajectoire), nous pouvons remonter de l'état latent $x_T$ jusqu'à la distribution des données :

\begin{equation}
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} | x_t)
\end{equation}

En conditionnant le modèle sur le pas de temps $t$, celui-ci va apprendre à prédire les paramètres gaussiens, à savoir la moyenne $\mu_\theta(x_t, t)$ et la matrice de covariance $\Sigma_\theta(x_t, t)$, pour chaque étape du processus. Mais alors, comment entraîner un tel modèle ?



\section{Implémentation d'un DDPM}

Ayant présenté les principes de base des modèles de diffusion, et plus particulièrement des DDPM, nous pouvons désormais aborder leur implémentation. 

\subsection{Architecture du modèle}

Dans cette section, nous présentons une architecture de DDPM habituelle/standard, inspirée d'une implémentation proposée par \textit{T. Matsuzaki}. L'architecture du modèle de diffusion est composée de plusieurs éléments clés, notamment un U-Net, un encodage du pas de temps, des blocks ResNet et des blocks d'Attention.

\subsubsection{U-Net}

Tout d'abord, le modèle de diffusion est basé sur une architecture de type U-Net, qui est largement utilisée dans les tâches de segmentation d'images et de génération. Le U-Net se compose d'un encodeur et d'un décodeur, avec des connexions de saut (skip connections) entre les couches correspondantes de l'encodeur et du décodeur afin de préserver les détails spatiaux de l'image tout au long du processus de génération.

La figure \ref{fig:unet} illustre l'architecture générale du U-Net considéré.



\subsubsection{Encodage du Pas de Temps}

Il s'agit dans un premier temps d'encoder le pas de temps $t$. L'encodage du pas de temps est une étape cruciale, car il permet au modèle de diffusion de prendre en compte le niveau de bruit présent dans l'image à chaque étape du processus de génération. De la même façon que pour les DDPMs inconditionnels, nous utilisons un encodage sinusoïdal (positional encoding, noté $PE(t)$ dans la suite) pour représenter le pas de temps $t$. Cet encodage est ensuite transformé à l'aide d'un MLP (Linear + SiLU + Linear) pour obtenir un vecteur de dimension 512, qui est ensuite injecté dans les différents blocks de ResNet du modèle. \\

Cet encodage permet au modèle d'apprendre à différencier les différentes étapes du processus de diffusion et à ajuster sa prédiction en conséquence. La figure \ref{fig:time_encoding} illustre l'encodage du pas de temps.\\

\begin{figure}[htbp]
\centering
    \resizebox{0.9\textwidth}{!}{
        \begin{tikzpicture}[node distance=0.8cm, every node/.style={draw, font=\scriptsize, fill=white}]
            \node[circle, fill=gray!10] (t) {$t$};
            \node[rectangle, fill=orange!10, right=of t] (pe) {Encodage Sinusoïdal, $PE(t)$};
            \node[rectangle, fill=blue!5, right=of pe, align=center] (mlp) {MLP\\(Linear + SiLU + Linear)};
            \node[rectangle, rounded corners, fill=green!10, right=of mlp] (out) {Vecteur (512)};
            \draw[-{Stealth}, thick] (t) -- (pe); \draw[-{Stealth}, thick] (pe) -- (mlp); \draw[-{Stealth}, thick] (mlp) -- (out);
        \end{tikzpicture}
    }
    \caption{Schéma de l'encodage du pas de temps dans un DDPM conditionnel.}
    \label{fig:time_encoding}
\end{figure}

L'encodage positionnel du pas de temps est défini de la manière suivante :\\
\begin{equation}
PE(t)_{2i} = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad PE(t)_{2i+1} = \cos\left(\frac{t}{10000^{2i/d}}\right)
\label{eq:positional_encoding}
\end{equation}
où $d$ est la dimension du vecteur d'encodage (dans notre cas, 512), et $i$ est l'indice de la dimension. 

L'intérêt d'appliquer un MLP à cet encodage sinusoïdal est de permettre au modèle d'apprendre une représentation plus riche et adaptée du pas de temps, suffisamment grande pour être réduite (si nécessaire) et injectée dans les différentes "couches" du modèle de diffusion.\\


\subsubsection{Blocks de ResNet}

\subsection{Algorithmes : Entraînement et Inférence}
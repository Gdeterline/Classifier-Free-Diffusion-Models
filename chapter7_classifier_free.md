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

En termes d'implémentation, le DDPM conditionnel n'est pas très différent du DDPM inconditionnel. La principale différence par rapport à l'implémentation inconditionnelle réside dans la prise en compte du pas de temps $t$ et de la classe $y$ (ou de la condition $y$ plus généralement) dans l'architecture du modèle. Les deux types d'informations (pas de temps et classe) sont encodées et injectées dans les blocks de ResNet du modèle de diffusion, au sein des différentes "couches" du modèle. \\

\subsubsection{Encodage du Pas de Temps}

Il s'agit dans un premier temps d'encoder le pas de temps $t$. L'encodage du pas de temps est une étape cruciale, car il permet au modèle de diffusion de prendre en compte le niveau de bruit présent dans l'image à chaque étape du processus de génération. De la même façon que pour les DDPMs inconditionnels, nous utilisons un encodage sinusoïdal (positional encoding, noté $PE(t)$ dans la suite) pour représenter le pas de temps $t$. Cet encodage est ensuite transformé à l'aide d'un MLP (Linear + SiLU + Linear) pour obtenir un vecteur de dimension 512, qui est ensuite injecté dans les différents blocks de ResNet du modèle. \\

Cet encodage permet au modèle d'apprendre à différencier les différentes étapes du processus de diffusion et à ajuster sa prédiction en conséquence. La figure \ref{fig:time_encoding} illustre l'encodage du pas de temps.\\

\begin{center}
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
\end{center}

L'encodage positionnel du pas de temps est défini de la manière suivante :\\
\begin{equation}
PE(t)_{2i} = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad PE(t)_{2i+1} = \cos\left(\frac{t}{10000^{2i/d}}\right)
\label{eq:positional_encoding}
\end{equation}
où $d$ est la dimension du vecteur d'encodage (dans notre cas, 512), et $i$ est l'indice de la dimension. 

L'intérêt d'appliquer un MLP à cet encodage sinusoïdal est de permettre au modèle d'apprendre une représentation plus riche et adaptée du pas de temps, suffisamment grande pour être réduite (si nécessaire) et injectée dans les différentes "couches" du modèle de diffusion.\\


\subsubsection{Encodage de la Classe}

L'encodage de la classe (ou de la condition $y$ plus généralement), quant à lui, est réalisé suivant une table de correspondance, qui associe à chaque classe un vecteur d'embedding de dimension 512. Par exemple, pour le dataset MNIST, qui comporte 10 classes (les chiffres de 0 à 9), nous avons une table d'embeddings de taille $10 \times 512$. Lors de l'entraînement, pour les exemples conditionnels, nous récupérons l'embedding correspondant à la classe $y$ associée à l'image, tandis que pour les exemples inconditionnels, nous utilisons un token spécial (un vecteur nul) pour représenter l'absence de conditionnement.\\

La figure \ref{fig:class_encoding} illustre l'encodage de la classe.\\

\begin{center}
    \resizebox{0.8\textwidth}{!}{
        \begin{tikzpicture}[node distance=1cm, every node/.style={draw, font=\scriptsize}]
            \node[fill=gray!10] (y) {$y$ (classe)};
            \node[fill=blue!5, right=of y] (emb) {Table d'embeddings\\nn.Embedding (10, 512)};
            \node[diamond, fill=red!5, right=of emb, aspect=1.5] (drop) {Dropout de classe\\P(dropout)=0.2};
            \node[rounded corners, fill=green!10, right=of drop] (out) {Vecteur $y_{emb}$ (512)};
            \draw[-{Stealth}, thick] (y) -- (emb); \draw[-{Stealth}, thick] (emb) -- (drop); \draw[-{Stealth}, thick] (drop) -- (out);
        \end{tikzpicture}
    }
    \caption{Schéma de l'encodage de la classe dans un DDPM conditionnel.}
    \label{fig:class_encoding}
\end{center}

Nous avons alors un vecteur d'embedding de classe $y_{emb}$ de dimension 512, qui est injecté dans les différentes "couches" du modèle de diffusion, de la même manière que le vecteur d'encodage du pas de temps.\\

\underline{Note:} Notons que dans notre implémentation, nous avons choisi d'appliquer un dropout de classe avec une probabilité de 0.2, ce qui signifie que 20\% des exemples présentés au modèle pendant l'entraînement sont inconditionnels (avec un token nul pour la classe), tandis que les 80\% restants sont conditionnels (avec la classe associée). Avec un dataset de taille suffisante, ce ratio permet au modèle d'apprendre à générer des images réalistes à la fois avec et sans conditionnement, ce qui est essentiel pour le bon fonctionnement de la Classifier-Free Guidance.\\

\underline{Note 2:} Précisons que les paramètres de la table de correspondance des classes (les embeddings) sont appris conjointement avec les autres paramètres du modèle de diffusion pendant l'entraînement, ce qui permet au modèle d'apprendre des représentations de classe adaptées à la tâche de génération.\\

\subsubsection{Architecture d'un block de ResNet conditionnel}

L'architecture d'un block de ResNet conditionnel est similaire à celle d'un block de ResNet inconditionnel, avec l'ajout de mécanismes pour intégrer les informations du pas de temps et de la classe.\\

La figure \ref{fig:resnet_block_cfg} illustre l'architecture d'un block de ResNet conditionnel. Nous avons les mêmes composantes que pour un block de ResNet inconditionnel (convolutions, normalisation, activation), mais avec la notion de modulation des canaux pour intégrer les informations du pas de temps et de la classe.\\

\begin{center}
    \resizebox{!}{0.7\paperheight}{
    \begin{tikzpicture}[node distance=0.5cm, every node/.style={font=\scriptsize, thick}]
        % --- CHEMIN PRINCIPAL ---
        \node (in) {Entrée $x$ ($C_{in}$)};
        \node[draw, fill=blue!5, below=0.3cm of in, minimum width=3cm] (n1) {GroupNorm + SiLU};
        \node[draw, fill=blue!10, below=of n1, minimum width=3cm] (c1) {Conv2D (3x3)};
        
        % Zone d'injection
        \node[draw, circle, fill=yellow!30, below=0.7cm of c1] (mult) {$\times$};
        \node[draw, circle, fill=yellow!30, below=0.6cm of mult] (plus_emb) {+};
        
        \node[draw, fill=blue!5, below=0.7cm of plus_emb, minimum width=3cm] (n2) {GroupNorm + SiLU + Dropout};
        \node[draw, fill=blue!10, below=of n2, minimum width=3cm] (c2) {Conv2D (3x3)};
        
        \node[draw, circle, fill=orange!20, below=0.6cm of c2] (plus_res) {+};
        \node[below=0.3cm of plus_res] (out) {Sortie ($C_{out}$)};

        % --- CONDITIONNEMENT (À droite) ---
        \node[draw, fill=orange!5, right=0.8cm of mult, align=left, font=\tiny, text width=1.5cm] (t_prj) {Proj. Temps\\(SiLU + Linear + Proj)};
        \node[draw, fill=green!5, right=0.8cm of plus_emb, align=left, font=\tiny, text width=1.5cm] (y_prj) {Proj. Classe\\(SiLU + Linear + Proj)};
        
        \draw[<-] (t_prj.east) -- ++(0.3,0) node[right, font=\tiny] {$t_{emb}$};
        \draw[<-] (y_prj.east) -- ++(0.3,0) node[right, font=\tiny] {$y_{emb}$};

        % --- CHEMIN RÉSIDUEL (À gauche) ---
        \node[draw, fill=gray!10, left=1.2cm of mult, font=\tiny, text width=2.2cm, align=center] (res_proj) {
            \begin{tabular}{c} 
                Skip Connection \\ 
                (Linear Projection) 
            \end{tabular}
        };

        % --- FLÈCHES CHEMIN PRINCIPAL ---
        \draw[-{Stealth}] (in) -- (n1);
        \draw[-{Stealth}] (n1) -- (c1);
        \draw[-{Stealth}] (c1) -- (mult);
        \draw[-{Stealth}] (mult) -- (plus_emb);
        \draw[-{Stealth}] (plus_emb) -- (n2);
        \draw[-{Stealth}] (n2) -- (c2);
        \draw[-{Stealth}] (c2) -- (plus_res);
        \draw[-{Stealth}] (plus_res) -- (out);

        % --- FLÈCHES CONDITIONNEMENT ---
        \draw[-{Stealth}] (t_prj.west) -- (mult.east) node[midway, above, font=\tiny] {Scale};
        \draw[-{Stealth}] (y_prj.west) -- (plus_emb.east) node[midway, above, font=\tiny] {Shift};

        % --- FLÈCHE RÉSIDUELLE (SKIP CONNECTION) ---
        % 1. Part de la GAUCHE (west) de x
        % 2. Descend et contourne pour arriver sur le haut du bloc gris (res_proj)
        % 3. Sort par le bas (south) vers le bloc d'addition finale
        \draw[-{Stealth}] (in.west) -- ++(-1.825,0) -- (res_proj.north);
        \draw[-{Stealth}] (res_proj.south) |- (plus_res.west);
    \end{tikzpicture}
    }
    \caption{Architecture d'un block de ResNet conditionnel pour un DDPM avec Classifier-Free Guidance.}
    \label{fig:resnet_block_cfg}
\end{center}

Si le fonctionnement du block de ResNet conditionnel est similaire à celui d'un block de ResNet inconditionnel, l'ajout de mécanismes pour intégrer les informations du pas de temps et de la classe (équation \ref{eq:resnet_modulation}) permet au modèle de moduler les canaux de manière conditionnelle, de manière à extraire des caractéristiques spécifiques à la classe et au niveau de bruit présent dans l'image à chaque étape du processus de diffusion.\\

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

\underline{Note:} Notons plusieurs aspects importants concernant l'architecture du block de ResNet en figure \ref{fig:resnet_block_cfg} :
\begin{itemize}
    \item Nous avons une skip connection (chemin résiduel) qui permet de faire passer l'entrée $x$ directement à la sortie du block, ce qui stabilise l'entraînement.
    \item L'utilisation de GroupNorm permet de diviser les canaux en groupes pour la normalisation, et est particulièrement adaptée dans la mesure où nous travaillons sur les canaux de l'image.
    \item Le dropout présent dans la deuxième partie du block de ResNet permet de régulariser le modèle et n'a pas d'influence sur la classe des images générées.
\end{itemize}

\subsection{Algorithmes : Entraînement et Inférence}

Ayant présenté l'architecture d'un block de ResNet conditionnel, nous pouvons maintenant détailler les algorithmes d'entraînement et d'inférence pour un DDPM avec Classifier-Free Guidance.\\

\begin{figure}[htbp]
\centering
\begin{minipage}[t]{0.48\textwidth}
\captionsetup{type=algorithm}
\caption{Entraînement DDPM + CFG}
\label{alg:cfg_training}
\begin{algorithmic}[1]
\Require Dataset $\mathcal{D}$, modèle $\epsilon_\theta$, embedding $\mathbf{E}$, $T$, $p_{\text{uncond}}$
\State Construire $\alpha_t$, $\bar{\alpha}_t=\prod_{i=0}^{t}\alpha_i$
\For{chaque mini-lot $(x_0,y)$}
    \State Échantillonner $t\sim\mathcal{U}(\{0,\dots,T-1\})$, $\varepsilon\sim\mathcal{N}(0,\mathbf{I})$
    \State $x_t \gets \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon$
    \State $y_{\text{emb}} \gets \mathbf{E}[y]$
    \State $m\sim\mathrm{Bernoulli}(1-p_{\text{uncond}})$
    \State $y_{\text{emb}} \gets m\odot y_{\text{emb}}$ \Comment{dropout de classe}
    \State $\hat{\varepsilon}\gets \epsilon_\theta(x_t,t,y_{\text{emb}})$
    \State $\mathcal{L}\gets \|\hat{\varepsilon}-\varepsilon\|_2^2$
    \State Mise à jour de $(\theta,\mathbf{E})$ par Adam
\EndFor
\end{algorithmic}
\end{minipage}
\hfill
\begin{minipage}[t]{0.48\textwidth}
\captionsetup{type=algorithm}
\caption{Inférence DDPM + CFG}
\label{alg:cfg_inference}
\begin{algorithmic}[1]
\Require Modèle entraîné $\epsilon_\theta$, embedding $\mathbf{E}$, classe $y$, guidance $s$, $T$
\State Construire $\alpha_t$, $\bar{\alpha}_t$
\State Calculer $\sigma_t^2=\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$, avec $\bar{\alpha}_{-1}=1$
\State $x_T\sim\mathcal{N}(0,\mathbf{I})$
\State $y_{\text{cond}}\gets \mathbf{E}[y]$, \quad $y_{\varnothing}\gets \mathbf{0}$
\For{$t=T-1$ \textbf{to} $0$}
    \State $\hat{\varepsilon}_{c}\gets \epsilon_\theta(x_t,t,y_{\text{cond}})$
    \State $\hat{\varepsilon}_{u}\gets \epsilon_\theta(x_t,t,y_{\varnothing})$
    \State $\hat{\varepsilon}\gets (1+s)\hat{\varepsilon}_{c}-s\hat{\varepsilon}_{u}$
    \If{$t>0$}
        \State $z\sim\mathcal{N}(0,\mathbf{I})$
    \Else
        \State $z\gets \mathbf{0}$
    \EndIf
    \State $x_{t-1}\gets \frac{1}{\sqrt{\alpha_t}}\!\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\varepsilon}\right)+\sigma_t z$
\EndFor
\State \Return $x_0$
\end{algorithmic}
\end{minipage}
\caption{Pseudo-code de l'entraînement et de l'inférence pour la Classifier-Free Guidance.}
\label{fig:cfg_algorithms_side_by_side}
\end{figure}

\section{Résultats expérimentaux}

\subsection{Qualité de la génération inconditionnelle/conditionnelle sur MNIST}
\subsection{Impact de l'échelle de \textit{Guidance}}






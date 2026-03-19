\chapter{Modèles de Diffusion}
\label{chap:intro_modeles}
\rhead{Modèles de Diffusion}
\section{Modèles de Diffusion}
Les modèles de diffusion constituent une nouvelle classe de modèles génératifs à l'état de l'art, capables de produire des images de haute résolution. L'idée principale de ces modèles repose sur la correction itérative de la prédiction à chaque pas de temps, jusqu'à l'obtention d'une génération finale cohérente. 

Dans cette partie, nous allons aborder les principes de base ainsi que l'architecture de ces modèles. \textbf{Pour l'élaboration de ce chapitre, nous nous concentrerons particulièrement sur les modèles probabilistes de diffusion par débruitage (\textit{Denoising Diffusion Probabilistic Models}, DDPM)} proposés par \cite{ho2022}, tout en notant qu'il existe d'autres approches basées sur la fonction de score (\textit{score-based models}).
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
\subsection{Entraînement d'un modèle de diffusion}

Si nous prenons un peu de recul, nous pouvons remarquer que la combinaison de $q$ et $p$ est très similaire à celle d'un auto-encodeur variationnel (VAE). Par conséquent, nous pouvons l'entraîner en optimisant la log-vraisemblance négative des données d'entraînement. Après une série de calculs que nous ne détaillerons pas ici, nous pouvons écrire la borne inférieure de l'évidence (\textit{Evidence Lower Bound} ou ELBO) comme suit :

\begin{equation}
\log p(x) \geq \mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)] - D_{KL}(q(x_T|x_0) || p(x_T)) - \sum_{t=2}^T \mathbb{E}_{q(x_t|x_0)} [D_{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t))]
\end{equation}

Cette expression peut être simplifiée sous la forme suivante :
\begin{equation}
\mathcal{L} = L_0 - L_T - \sum_{t=2}^T L_{t-1}
\end{equation}

L'analyse des termes de l'ELBO permet de mieux comprendre les objectifs du modèle :
\begin{itemize}
    \item \textbf{Reconstruction :} Le terme $\mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)]$ est un terme de reconstruction similaire à celui d'un VAE.
    \item \textbf{Prior matching :} $D_{KL}(q(x_T|x_0) || p(x_T))$ mesure la proximité de la distribution finale avec une gaussienne standard. Sans paramètres entraînés, il est ignoré lors de l'apprentissage.
    \item \textbf{Dénoyautage (Denoising) :} Le terme $\sum_{t=2}^T L_{t-1}$ représente l'écart entre les étapes de débruitage réelles et celles prédites par le modèle.
\end{itemize}

\subsubsection{Rendre le processus inverse traitable}

Bien que $q(x_{t-1}|x_t)$ soit intraitable, le conditionnement sur $x_0$ permet d'obtenir une forme fermée :
\begin{equation}
q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1} ; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I})
\end{equation}
Grâce à l'astuce de reparamétrage, nous pouvons exprimer $x_0$ en fonction de $x_t$ et du bruit $\epsilon$. En substituant cette expression, la moyenne cible $\tilde{\mu}_t$ devient dépendante de $\epsilon$ :
\begin{equation}
\tilde{\mu}_t(x_t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon \right)
\end{equation}

\subsubsection{Prédire le bruit : La perte simplifiée}

Cette formulation montre qu'au lieu de prédire la moyenne de la distribution, le modèle peut simplement apprendre à prédire le bruit $\epsilon$ ajouté à chaque étape.\cite{Ho2020} ont proposé une version simplifiée de la fonction de perte qui surpasse l'objectif théorique :

\begin{equation}
L_{simple}(\theta) = \mathbb{E}_{x_0, t, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) \right\|^2 \right]
\end{equation}

Dans le modèle DDPM original \cite{Ho2020}, la variance $\Sigma_\theta$ est maintenue fixe, et le réseau n'apprend que la moyenne.

% Transition vers implémentation


\section{Implémentation d'un DDPM}

Ayant présenté les principes de base des modèles de diffusion, et plus particulièrement des DDPM, nous pouvons désormais aborder leur implémentation.\\

\subsection{Architecture du modèle}

Dans cette section, nous présentons une architecture de DDPM standard, issue de l'implémentation proposée par \textit{Jonathan Ho, Ajay Jain, Pieter Abbeel}. L'architecture du modèle de diffusion est composée de plusieurs éléments clés, notamment un U-Net, un encodage du pas de temps, des blocks ResNet et des blocks d'Attention.\\


\subsubsection{U-Net}

Tout d'abord, le modèle de diffusion est basé sur une architecture de type U-Net, qui est largement utilisée dans les tâches de segmentation d'images et de génération. Le U-Net se compose d'un encodeur et d'un décodeur, avec des connexions de saut (skip connections) entre les couches correspondantes de l'encodeur et du décodeur afin de préserver les détails spatiaux de l'image tout au long du processus de génération.\\

La figure \ref{fig:unet} illustre l'architecture générale du U-Net considéré.

\begin{figure}[htbp]
    \centering
    \resizebox{!}{0.4\paperheight}{
        \begin{tikzpicture}[
            node distance=0.4cm, 
            every node/.style={font=\tiny, thick},
            % Définition des styles
            res/.style={draw, rectangle, fill=blue!10, minimum width=2.5cm, minimum height=0.4cm},
            att/.style={draw, rectangle, fill=red!10, minimum width=2.5cm, minimum height=0.4cm},
            down/.style={draw, rectangle, fill=gray!20, minimum width=2.5cm},
            up/.style={draw, rectangle, fill=orange!20, minimum width=2.5cm},
            time/.style={inner sep=2pt, font=\tiny\bfseries, color=blue!70!black}
        ]

            % --- ENCODER ---
            \node[draw, fill=green!5] (in) {Entrée $x_t$};
            \node[res, below=0.2cm of in] (e1) {2x ResNet};
            \node[down, below=of e1] (d1) {DownSample};
            
            \node[att, below=0.4cm of d1] (e2) {2x ResNet + \textbf{Attn}};
            \node[down, below=of e2] (d2) {DownSample};
            
            \node[res, below=0.4cm of d2] (e3) {2x ResNet};
            \node[down, below=of e3] (d3) {DownSample};
            
            \node[res, below=0.4cm of d3] (e4) {2x ResNet};

            % --- MIDDLE (Bottleneck) ---
            \node[att, below=0.8cm of e4, xshift=2.25cm, text width=3cm, align=center, fill=purple!10] (mid) {
                \textbf{Bottleneck}\\ResNet $\to$ \textbf{Attn} $\to$ ResNet
            };

            % --- DECODER ---
            \node[res, right=2cm of e4] (u4) {3x ResNet};
            
            \node[up, above=0.4cm of u4] (up3) {UpSample};
            \node[res, above=of up3] (u3) {3x ResNet};
            
            \node[up, above=0.4cm of u3] (up2) {UpSample};
            \node[att, above=of up2] (u2) {3x ResNet + Attention};
            
            \node[up, above=0.4cm of u2] (up1) {UpSample};
            \node[res, above=of up1] (u1) {3x ResNet};
            
            \node[draw, fill=green!5, above=0.2cm of u1] (out) {Sortie $\epsilon_\theta(x_t, t)$};

            % --- INJECTION DU TEMPS (t) ---
            % Encodeur
            \foreach \n in {e1, e2, e3, e4} {
                \node (t_\n) [left=0.4cm of \n, time] {$t$};
                \draw[->, blue!50, thin] (t_\n) -- (\n);
            }
            % Middle
            \node (t_mid) [below=0.2cm of mid, time] {$t$};
            \draw[->, blue!50, thin] (t_mid) -- (mid);
            
            % Décodeur
            \foreach \n in {u1, u2, u3, u4} {
                \node (t_\n) [right=0.4cm of \n, time] {$t$};
                \draw[->, blue!50, thin] (t_\n) -- (\n);
            }

            % --- SKIP CONNECTIONS ---
            \draw[dashed, red, ->] (e1.east) -- node[above, font=\tiny, color=black, pos=0.5] {cat} (u1.west);
            \draw[dashed, red, ->] (e2.east) -- (u2.west);
            \draw[dashed, red, ->] (e3.east) -- (u3.west);
            \draw[dashed, red, ->] (e4.east) -- (u4.west);

            % --- FLUX PRINCIPAL ---
            \draw[->] (in) -- (e1); \draw[->] (e1) -- (d1); \draw[->] (d1) -- (e2); \draw[->] (e2) -- (d2);
            \draw[->] (d2) -- (e3); \draw[->] (e3) -- (d3); \draw[->] (d3) -- (e4); 
            \draw[->] (e4.south) -- ++(0,-1.26) -- (mid.west);
            \draw[->] (mid.east) -- ++(0.66,0) -- (u4.south);
            \draw[->] (u4) -- (up3); \draw[->] (up3) -- (u3); \draw[->] (u3) -- (up2);
            \draw[->] (up2) -- (u2); \draw[->] (u2) -- (up1); \draw[->] (up1) -- (u1); \draw[->] (u1) -- (out);
            
        \end{tikzpicture}%
    }
    \caption{Architecture détaillée du U-Net avec injection du Time Embedding ($t$) dans les blocs ResNet.}
    \label{fig:unet}
\end{figure}

Nous pouvons observer que le modèle prend en entrée une image bruitée $x_t$, et qu'à chaque étape du processus de diffusion, il reçoit également le pas de temps $t$, correspondant à l'étape actuelle du processus de diffusion, injecté dans les différents blocks de ResNet du modèle.\\

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
où $d$ est la dimension du vecteur d'encodage (dans notre cas, 512), et $i$ est l'indice de la dimension.\\

L'intérêt d'appliquer un MLP à cet encodage sinusoïdal est de permettre au modèle d'apprendre une représentation plus riche et adaptée du pas de temps, suffisamment grande pour être réduite (si nécessaire) et injectée dans les différentes "couches" du modèle de diffusion.\\

\subsubsection{Blocks ResNet}

Afin de prendre en compte le pas de temps dans les différentes étapes du processus de diffusion, on applique une fonction $SiLU$ (Sigmoid Linear Unit) à la sortie du MLP, puis une couche linéaire finale pour projeter ce vecteur d'encodage du pas de temps à la dimension des canaux du modèle de diffusion (par exemple, 128 ou 256 canaux). Alors, ce vecteur projeté (noté $t_{prj}$) est injecté dans les différents blocks de ResNet du modèle. Cette injection se fait généralement par addition (c'est-à-dire suivant la formule $out = out + t_prj$), ce qui permet au modèle de diffusion d'apprendre à ajuster sa prédiction en fonction du niveau de bruit présent dans l'image à chaque étape du processus de génération.\\

La figure \ref{fig:resnet_block_ddpm} illustre un block de ResNet typique utilisé dans le modèle de diffusion, avec l'injection du pas de temps projeté.\\

\begin{figure}[htbp]
\centering
    \resizebox{!}{0.3\paperheight}{
    \begin{tikzpicture}[node distance=0.5cm, every node/.style={font=\scriptsize, thick}]
        % --- CHEMIN PRINCIPAL ---
        \node (in) {Entrée $x$ ($C_{in}$)};
        \node[draw, fill=blue!5, below=0.3cm of in, minimum width=3cm] (n1) {GroupNorm + SiLU};
        \node[draw, fill=blue!10, below=of n1, minimum width=3cm] (c1) {Conv2D (3x3)};
        
        % Zone d'injection
        \node[draw, circle, fill=yellow!30, below=0.6cm of c1] (plus_emb) {+};
        
        \node[draw, fill=blue!5, below=0.7cm of plus_emb, minimum width=3cm] (n2) {GroupNorm + SiLU + Dropout};
        \node[draw, fill=blue!10, below=of n2, minimum width=3cm] (c2) {Conv2D (3x3)};
        
        \node[draw, circle, fill=orange!20, below=0.6cm of c2] (plus_res) {+};
        \node[below=0.3cm of plus_res] (out) {Sortie ($C_{out}$)};

        % --- CONDITIONNEMENT (À droite) ---
        \node[draw, fill=orange!5, right=0.8cm of plus_emb, align=left, font=\tiny, text width=1.5cm] (t_prj) {Proj. Temps\\(SiLU + Linear)};
        
        \draw[<-] (t_prj.east) -- ++(0.3,0) node[right, font=\tiny] {$t_{emb}$};

        % --- CHEMIN RÉSIDUEL (À gauche) ---
        \node[draw, fill=gray!10, left=1.2cm of plus_emb, font=\tiny, text width=2.5cm, align=center] (res_proj) {
            \begin{tabular}{c} 
                Skip Connection \\ 
                (Linear Projection) 
            \end{tabular}
        };

        % --- FLÈCHES CHEMIN PRINCIPAL ---
        \draw[-{Stealth}] (in) -- (n1);
        \draw[-{Stealth}] (n1) -- (c1);
        \draw[-{Stealth}] (c1) -- (plus_emb);
        \draw[-{Stealth}] (plus_emb) -- (n2);
        \draw[-{Stealth}] (n2) -- (c2);
        \draw[-{Stealth}] (c2) -- (plus_res);
        \draw[-{Stealth}] (plus_res) -- (out);

        % --- FLÈCHES CONDITIONNEMENT ---
        \draw[-{Stealth}] (t_prj.west) -- (plus_emb.east) node[midway, above, font=\tiny] {Shift};

        % --- FLÈCHE RÉSIDUELLE (SKIP CONNECTION) ---
        % 1. Part de la GAUCHE (west) de x
        % 2. Descend et contourne pour arriver sur le haut du bloc gris (res_proj)
        % 3. Sort par le bas (south) vers le bloc d'addition finale
        \draw[-{Stealth}] (in.west) -- ++(-1.8,0) -- (res_proj.north);
        \draw[-{Stealth}] (res_proj.south) |- (plus_res.west);
    \end{tikzpicture}
    }
    \caption{Architecture d'un block de ResNet pour un DDPM.}
    \label{fig:resnet_block_ddpm}
\end{figure}


\subsubsection{Attention Blocks}

\subsection{Algorithmes : Entraînement et Inférence}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{images/training-sampling-ddpm.png}
        \caption*{Algorithmes d'entraînement et d'échantillonnage des DDPM.\cite{Ho2020}}
    \label{fig:ddpm_algos_combined}
\end{figure}
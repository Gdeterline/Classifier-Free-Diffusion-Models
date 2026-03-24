\chapter{Modèles de Diffusion}
\label{chap:intro_modeles}
\rhead{Modèles de Diffusion}
\section{Modèles de Diffusion}

Les modèles de diffusion constituent une nouvelle classe de modèles génératifs à l'état de l'art, capables de produire des images de haute résolution. L'idée principale de ces modèles repose sur la correction itérative de la prédiction à chaque pas de temps, jusqu'à l'obtention d'une génération finale cohérente. \\

Dans cette partie, nous allons aborder les principes généraux ainsi que l'architecture de ces modèles. \textbf{Dans la suite de l'étude, nous nous intéresserons particulièrement aux modèles probabilistes de diffusion par débruitage (\textit{Denoising Diffusion Probabilistic Models}, DDPM)} proposés par \textit{Ho et al.} \cite{Ho2020}, tout en notant qu'il existe d'autres approches basées sur la fonction de score (\textit{score-based models}).\\

Avant d'aborder les DDPM, nous allons expliquer brièvement les modèles à base de score et leur lien avec les modèles de diffusion.
\subsection{Modèles de diffusion basées sur la fonction de score}
\noindent Supposons que nous disposions d'un ensemble de données $\mathcal{D} = \{x_1, x_2, \dots, x_n\}$ où chaque échantillon est issu d'une distribution multimodale. Par exemple, le jeu de données MNIST regroupe des chiffres de 0 à 9, où chaque $x_i$ représente une classe spécifique. L'objectif est de concevoir un \textbf{modèle génératif} capable de modéliser l'intégralité de cette distribution d'images. Une fois le modèle entraîné, nous pourrons synthétiser de nouvelles images par \textbf{échantillonnage} à partir de la distribution apprise.

Pour construire un modèle génératif, nous devons d'abord trouver une façon de poser le problème, notamment en utilisant un modèle basé sur la vraisemblance pour modéliser directement la densité de probabilité. Considérons une fonction $f_\theta(\mathbf{x})$, paramétrée par un paramètre apprenable $\theta$. Nous pouvons définir une fonction de densité de probabilité (pdf) via l'expression :
\begin{equation}
    p_\theta(\mathbf{x}) = \frac{e^{-f_\theta(\mathbf{x})}}{Z_\theta}
    \label{zz}
\end{equation}
où $Z_\theta$ est un paramètre de normalisation dépendant de $\theta$, qui assure que $p_\theta$ estimée soit une distribution de probabilité telle que $\int p_\theta(\mathbf{x}) d\mathbf{x} = 1$. Ici, $f_\theta$ représente notre réseau de neurones. Nous pouvons alors assurer l'apprentissage de $p_\theta(\mathbf{x})$ en maximisant la log-vraisemblance des données selon la formulation :
\begin{equation}
    \max_\theta \sum_{i=1}^N \log p_\theta(\mathbf{x}_i)
    \label{dd}
\end{equation}
Cependant, l'équation (\ref{dd}) impose que $p_\theta(\mathbf{x})$ soit une fonction de densité de probabilité. Pour calculer $p_\theta(\mathbf{x})$, il est donc nécessaire d'évaluer la constante de normalisation $Z_\theta$, une quantité généralement incalculable pour une fonction $f_\theta(\mathbf{x})$ générale. 

Pour contourner cette difficulté, l'approche proposée par \cite{DBLP:journals/corr/abs-1907-05600} repose sur l'estimation de la fonction de score $s_{\theta}(x)=\nabla_{\mathbf{x}} \log p(\mathbf{x})$. Le modèle de score peut être paramétré sans tenir compte de la constante de normalisation $Z_\theta$ en appliquant le gradient sur l'équation (\ref{zz}) selon le développement suivant :

\begin{align}
    s_\theta(\mathbf{x}) &= \nabla_{\mathbf{x}} \log p_\theta(\mathbf{x}) \\
    &= \nabla_{\mathbf{x}} \log \left( \frac{e^{-f_\theta(\mathbf{x})}}{Z_\theta} \right) \\
    &= \nabla_{\mathbf{x}} \left[ -f_\theta(\mathbf{x}) - \log Z_\theta \right] \\
    &= -\nabla_{\mathbf{x}} f_\theta(\mathbf{x}) - \underbrace{\nabla_{\mathbf{x}} \log Z_\theta}_{0} \\
    s_\theta(\mathbf{x}) &= -\nabla_{\mathbf{x}} f_\theta(\mathbf{x})
\end{align}

Comme $Z_\theta$ ne dépend pas de $\mathbf{x}$, son gradient est nul, ce qui permet d'apprendre le score directement via le réseau de neurones $f_\theta$.

De manière analogue aux modèles basés sur la vraisemblance, nous pouvons entraîner des modèles basés sur le score en minimisant la divergence de Fisher entre la distribution du modèle et celle des données, définie par l'expression :
\begin{equation}
    \mathbb{E}_{p(\mathbf{x})} \left[ \| \nabla_{\mathbf{x}} \log p(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x}) \|_2^2 \right]
\end{equation}
Intuitivement, cette divergence compare le carré de la distance $\ell_2$ entre le score réel des données (\textit{ground-truth}) et le modèle basé sur le score. Cependant, le calcul direct de cette mesure est irréalisable car il nécessite l'accès au score inconnu des données, $\nabla_{\mathbf{x}} \log p(\mathbf{x})$. Heureusement, il existe une famille de méthodes appelées \textit{score matching} qui permettent de minimiser la divergence de Fisher sans connaître explicitement le score réel des données. Pour plus de détails sur la transformation mathématique permettant de s'affranchir du score inconnu via l'intégration par parties, on pourra se référer à l'annexe \ref{annexe:score_matching}. Ces objectifs de \textit{score matching} peuvent être directement estimés sur un jeu de données et optimisés par descente de gradient stochastique, de manière analogue à l'objectif de log-vraisemblance utilisé pour l'entraînement des modèles classiques.\\

\subsection{Relation entre DDPM et modèles basés sur le score}

Avant d'aborder les modèles de diffusion probabilistes (DDPM), il est essentiel d'établir le lien théorique avec les modèles basés sur le score (\textit{Score-based Generative Models}). Dans l'approche par le score, on entraîne un réseau de neurones à prédire le gradient de la log-densité de probabilité, noté $\nabla_{\mathbf{x}} \log p_{\theta}(\mathbf{x})$, qui indique la direction vers les régions de haute densité de la distribution des données. 

Dans un processus de diffusion gaussien (DDPM), du bruit est injecté successivement jusqu'à obtenir une image totalement bruitée. On entraîne alors un réseau à prédire le bruit $\mathbf{\epsilon}$ ajouté à chaque pas de temps $t$. Mathématiquement, cette prédiction est équivalente au score de la distribution perturbée selon la relation :
\begin{equation}
    \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | \mathbf{x}_0) = - \frac{\mathbf{\epsilon}}{\sigma_t}
\end{equation}
Ainsi, prédire le bruit $\mathbf{\epsilon}$ (approche DDPM) revient strictement à estimer le score (approche \textit{Score Matching}), à un facteur d'échelle près ($1/\sigma_t$).

L'utilisation prédominante des DDPM s'explique principalement par la simplicité de leur entraînement. La fonction de perte est simplifiée sous la forme d'une distance $L_2$ entre le bruit réel injecté et le bruit prédit par le modèle :
\begin{equation}
    L_{\text{simple}} = \mathbb{E} \left[ \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 \right]
\end{equation}
De plus, les modèles basés sur le score classiques souffrent souvent dans les régions de faible densité où le gradient est quasi nul, empêchant toute convergence. L'ajout successif de bruit dans le cadre des DDPM permet d'étaler les données sur tout l'espace de configuration. Le modèle dispose ainsi d'échantillons sur l'ensemble du domaine pour apprendre efficacement le score, représenté ici par le bruit.
\newpage
\section{Denoising Diffusion Probabilistic Models (DDPM)}

Nous proposons à présent d'orienter notre étude vers les modèles de diffusion par débruitage, plus précisément les Denoising Diffusion Probabilistic Models (DDPM) proposés par \textit{Ho et al.} \cite{Ho2020}. Ces modèles sont basés sur un processus de diffusion directe, dans lequel du bruit est ajouté progressivement à un échantillon de données, et un processus de diffusion inverse, dans lequel le modèle apprend à inverser ce processus de diffusion pour générer de nouvelles données à partir d'un échantillon bruité.\\

\subsection{Processus de diffusion directe (Forward Process)}

Afin de construire un modèle de diffusion, nous devons d'abord définir un processus de diffusion directe, qui consiste à ajouter du bruit à un échantillon de données à chaque étape $t$ de la chaîne.\\

Supposons que nous disposions d'un point de données $x_0$ échantillonné à partir de la distribution réelle $q(x)$ ($x_0 \sim q(x)$). Nous pouvons définir un processus de diffusion directe en ajoutant progressivement du bruit à chaque étape de la chaîne. Plus précisément, à chaque itération $t$, on injecte un bruit gaussien de variance $\beta_t$ à l'échantillon $x_{t-1}$, produisant ainsi une nouvelle variable  $x_t$. La distribution de transition $q(x_t | x_{t-1})$ de ce processus peut être formulée comme suit :\\

\begin{equation}
q(x_t | x_{t-1}) = \mathcal{N}(x_t ; \mu_t = \sqrt{1 - \beta_t} x_{t-1}, \Sigma_t = \beta_t \mathbf{I})
\end{equation}

\vspace{0.1cm}

où $\beta_t$ est un hyperparamètre qui contrôle la quantité de bruit ajoutée à chaque étape (on parle de \textit{variance schedule}, ou de \textit{planification de la variance}), et $\mathbf{I}$ est la matrice identité. \\

Nous pouvons exprimer $x_t$ directement en fonction de $x_0$ et du bruit ajouté à chaque étape. L'annexe \ref{annexe:diffusion_details} présente les détails de ce développement mathématique, qui conduit à l'expression suivante :\\

\begin{equation}
q(x_t | x_0) = \mathcal{N}(x_t ; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})
\label{eq:diffusion_directe_xt_x0}
\end{equation}

\vspace{0.1cm}

où $\alpha_t = 1 - \beta_t$ et $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$. Nous avons alors une expression explicite de la distribution de $x_t$ en fonction de $x_0$ et du bruit ajouté, ce qui facilite grandement les calculs ultérieurs pour le processus de diffusion inverse.\\

\underline{Note :} Afin de pouvoir écrire l'équation (\ref{eq:diffusion_directe_xt_x0}), nous devons faire l'hypothèse d'une chaîne de Markov, c'est-à-dire que $x_t$ ne dépend que de $x_{t-1}$ et pas des étapes précédentes. Cette hypothèse est raisonnable dans le contexte de la diffusion, car à chaque étape, nous ajoutons du bruit de manière indépendante, ce qui rend les étapes précédentes non informatives pour la distribution de $x_t$ une fois que nous connaissons $x_{t-1}$.\\

\subsection{Processus de diffusion inverse (\textit{Reverse Diffusion})}
\label{sec:diff_inv}
Lorsque $T \to \infty$, l'échantillon  $x_T$ devient quasiment une distribution gaussienne isotrope. Par conséquent, si nous parvenons à apprendre à inverser le processus direct, c'est-à-dire modéliser la distribution $q(x_{t-1} | x_t)$, nous pouvons échantillonner $x_T$ à partir de $\mathcal{N}(0, \mathbf{I})$, exécuter le processus inverse/génératif et obtenir un nouvel échantillon de $q(x_0)$. Nous pouvons alors générer un nouveau point de donnée issu de la distribution d'origine. Il s'agit donc de savoir comment modéliser ce processus de diffusion inverse.\\

\subsubsection{Approximation du processus inverse par un réseau de neurones}

En pratique, la distribution $q(x_{t-1} | x_t)$ est intraitable car son estimation statistique nécessiterait des calculs impliquant l'ensemble de la distribution de données $q(x_0)$. Par conséquent, nous l'approximons plutôt, par un modèle paramétré $p_\theta$ (par exemple, un réseau de neurones). \\

Pour des valeurs de $\beta_t$ suffisamment petites, la distribution $q(x_{t-1} | x_t)$ est également gaussienne, ce qui nous permet de choisir $p_\theta$ comme une distribution gaussienne et de paramétrer simplement sa moyenne et sa variance. Plus précisément, nous pouvons définir $p_\theta$ comme suit :\\

\begin{equation}
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1} ; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
\label{eq:diffusion_inverse}
\end{equation}

\vspace{0.1cm}

En appliquant la formule \ref{eq:diffusion_inverse} à chaque pas de temps $t$, nous pouvons remonter de l'état $x_T$ jusqu'à l'état $x_0$ et ainsi générer un échantillon $x_0$ issu de la distribution d'origine $q(x_0)$.\\

En conditionnant le modèle sur le pas de temps $t$, celui-ci va apprendre à prédire les paramètres des distributions gaussiennes, à savoir la moyenne $\mu_\theta(x_t, t)$ et la matrice de covariance $\Sigma_\theta(x_t, t)$, pour chaque étape du processus.\\

\subsection{Entraînement d'un modèle de diffusion}

Ayant défini les processus de diffusion directe et inverse, nous pouvons à présent aborder l'entraînement du modèle de diffusion.\\

Tout d'abord, remarquons que la combinaison de $q$ et $p$ est très similaire à celle d'un auto-encodeur variationnel (VAE). En effet, $q$ joue le rôle d'un encodeur qui "déconstruit" les données en ajoutant du bruit, tandis que $p_\theta$ agit comme un décodeur qui tente de reconstruire les données à partir de l'état bruité. Alors, de la même manière que pour les VAE, nous cherchons à entraîner le modèle de diffusion en maximisant la log-vraisemblance des données d'entraînement. Néanmoins, le calcul direct de la log-vraisemblance est intraitable (il nécessiterait d'intégrer toutes les trajectoires de bruit possibles), et plutôt que de maximiser directement la log-vraisemblance, nous cherchons à optimiser une borne inférieure de l'évidence (ELBO), qui est un objectif plus tractable à optimiser.\\

Après une série de calculs que nous ne détaillerons pas ici, nous pouvons écrire la borne inférieure de l'évidence (\textit{Evidence Lower Bound} ou ELBO) comme suit :\\

\begin{equation}
\log p(x) \geq \mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)] - D_{KL}(q(x_T|x_0) || p(x_T)) - \sum_{t=2}^T \mathbb{E}_{q(x_t|x_0)} [D_{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t))]
\end{equation}

\vspace{0.1cm}

L'analyse des termes de l'ELBO permet de mieux comprendre les objectifs du modèle :
\begin{itemize}
    \item \textbf{Reconstruction :} Le terme $\mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)]$ est un terme de reconstruction similaire à celui d'un VAE, qui encourage le modèle à apprendre à reconstruire les données d'origine à partir de l'état bruité $x_1$.
    \item \textbf{Prior matching :} $D_{KL}(q(x_T|x_0) || p(x_T))$ mesure la distance (au sens de Kullback-Leibler) entre la distribution de $x_T$ sachant $x_0$ et la distribution de $x_T$ du modèle (distribution gaussienne isotropique).
    \item \textbf{ Denoising:} Le terme $\sum_{t=2}^T L_{t-1}$ représente l'écart entre les étapes de débruitage réelles et celles prédites par le modèle.\\
\end{itemize}

\subsubsection{Rendre le processus inverse traitable}

Comme nous avions commencé à le mentionner dans la section \ref{sec:diff_inv}, le processus de diffusion inverse est mathématiquement intraitable, car il nécessite de calculer des intégrales impliquant la distribution de données $q(x_0)$. En revanche, nous pouvons calculer $q(x_{t-1} | x_t, x_0)$, qui est la distribution de $x_{t-1}$ sachant $x_t$ et $x_0$. En utilisant les propriétés des distributions gaussiennes, nous pouvons montrer que $q(x_{t-1} | x_t, x_0)$ est également une distribution gaussienne dont la moyenne et la variance peuvent être calculées de manière analytique. 

\begin{equation}
q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1} ; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I})
\label{eq:q_x_t-1_xt_x0}
\end{equation}

Les détails de ce développement mathématique ne sont pas donnés dans ce rapport, mais peuvent être trouvés dans l'article \textit{What are Diffusion Models?} de L. Weng \cite{weng2021}. Nous pouvons toutefois donner l'expression de la moyenne de $q(x_{t-1} | x_t, x_0)$: \\

\begin{equation}
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0
\label{eq:mu_q_x_t-1_xt_x0}
\end{equation}

Nous avions établi, dans l'expression \ref{eq:diffusion_directe_xt_x0}, que $x_t$ peut être exprimé directement en fonction de $x_0$ et du bruit ajouté à chaque étape. 

Nous avons: 

\begin{equation}
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
\label{eq:xt_x0_epsilon}
\end{equation}

où $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ représente le bruit ajouté à chaque étape. En réarrangeant l'équation (\ref{eq:xt_x0_epsilon}), nous pouvons exprimer $x_0$ en fonction de $x_t$ et du bruit $\epsilon$ :

\begin{equation}
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon \right)
\end{equation}

En substituant cette expression de $x_0$ dans la formule de la moyenne de $q(x_{t-1} | x_t, x_0)$ (équation \ref{eq:mu_q_x_t-1_xt_x0}), nous obtenons une formulation qui permet au modèle de prédire directement le bruit $\epsilon$ à partir de $x_t$ et du pas de temps $t$ :\\

\begin{equation}
\tilde{\mu}_t(x_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)
\label{eq:mu_q_x_t-1_xt_epsilon}
\end{equation}

\subsubsection{Prédire le bruit : La perte simplifiée}

Cette formulation montre qu'au lieu de prédire la moyenne de la distribution, le modèle peut simplement apprendre à prédire le bruit $\epsilon$ ajouté à chaque étape. \textit{Ho et al.} ont proposé une version simplifiée de la fonction de perte qui surpasse l'objectif théorique :

\begin{equation}
L_{simple}(\theta) = \mathbb{E}_{x_0, t, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) \right\|^2 \right]
\end{equation}

Dans le modèle DDPM original (\textit{Ho et al.}, \cite{Ho2020}), la variance $\Sigma_\theta$ est maintenue fixe, et le réseau n'apprend que la moyenne.\\

Ayant présenté les principes de base des modèles de diffusion, et plus particulièrement des DDPM, nous pouvons désormais aborder leur implémentation.\\

\section{Implémentation d'un DDPM}

L'implémentation d'un DDPM repose sur plusieurs éléments clés, notamment l'architecture du modèle, l'influence du pas de temps, les blocks de ResNet et d'Attention, ainsi que les algorithmes d'entraînement et d'inférence.\\

\subsection{Architecture du modèle}

Dans cette section, nous présentons une architecture de DDPM standard, issue de l'implémentation proposée par \textit{Jonathan Ho, Ajay Jain, Pieter Abbeel}. L'architecture du modèle de diffusion est composée de plusieurs éléments clés, notamment un U-Net, un encodage du pas de temps, des blocks ResNet et des blocks d'Attention.\\

\subsubsection{U-Net}

Tout d'abord, le modèle de diffusion est basé sur une architecture de type U-Net, qui est largement utilisée dans les tâches de segmentation d'images et de génération. Le U-Net se compose d'un encodeur et d'un décodeur, avec des connexions de saut (skip connections) entre les couches correspondantes de l'encodeur et du décodeur afin de préserver les détails spatiaux de l'image tout au long du processus de génération.\\

La figure \ref{fig:unet} illustre l'architecture générale du U-Net considéré. 

\begin{figure}[htbp]
    \centering
% On passe à 70% de la largeur du texte pour une taille plus raisonnable
    \resizebox{0.5\linewidth}{!}{
        \begin{tikzpicture}[
            node distance=0.3cm, % Réduction légère de l'espace entre blocs
            every node/.style={font=\tiny, thick},
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

L'encodage du pas de temps est une étape cruciale, car il permet au modèle de diffusion de prendre en compte le niveau de bruit présent dans l'image à chaque étape du processus de génération. En effet, le même réseau de neurones est utilisé pour prédire le bruit à chaque étape, et il doit donc être capable de différencier les différentes étapes du processus de diffusion pour ajuster sa prédiction en conséquence.
Nous utilisons un encodage sinusoïdal (positional encoding, noté $PE(t)$ dans la suite) pour représenter le pas de temps $t$. Cet encodage est ensuite transformé à l'aide d'un MLP (Linear + SiLU + Linear) pour obtenir un vecteur de dimension 512, qui est ensuite injecté dans les différents blocks de ResNet du modèle. \\

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

Afin de prendre en compte les différentes étapes du processus de diffusion, le modèle intègre des blocks de ResNet, dans lesquels le pas de temps est injecté.

Le pas de temps, initialement encodé en un vecteur de dimension 512, est projeté à la dimension des canaux du modèle de diffusion (par exemple, 128 ou 256 canaux). Ce vecteur, noté $t_{prj}$, est alors additionné à la sortie de la première couche de convolution du block de ResNet, suivant la formule \ref{eq:resnet_time_injection} :\\

\begin{equation}
\text{out} = \text{out} + t_{prj}
\label{eq:resnet_time_injection}
\end{equation}

Le modèle de diffusion est alors capable d'apprendre à ajuster sa prédiction en fonction du niveau de bruit présent dans l'image à chaque étape du processus de génération.\\

L'architecture complète d'un block de ResNet conditionnel pour un DDPM avec Classifier-Free Guidance est présentée en annexe \ref{annexe:cfg_resnet_architecture}, figure \ref{fig:resnet_block_ddpm}.\\

\subsubsection{Blocks d'Attention}

Si nous nous référons à la figure \ref{fig:unet}, nous pouvons observer qu'en plus des blocks de ResNet, le modèle de diffusion intègre également des blocks d'Attention à certains endroits stratégiques de l'architecture (notamment dans les étapes intermédiaires du U-Net). Ces blocks d'Attention permettent au modèle de diffusion de capturer les dépendances à long terme dans l'image, ce qui est crucial pour générer des images cohérentes et de haute qualité.\\

Les blocs d'Attention utilisés dans les modèles de diffusion sont basés sur ceux proposés par \textit{Vaswani et al.} dans leur article "Attention is All You Need" \cite{Vaswani2017}, mais adaptés pour fonctionner dans le contexte de la génération d'images. Ces blocks d'Attention permettent au modèle de diffusion de se concentrer sur différentes parties de l'image à chaque étape du processus de génération, ce qui améliore la qualité des images générées.
Notons que l'architecture du modèle de diffusion ne contient qu'un nombre restreint de blocs d'Attention. Ceux-ci ne sont pas placés aux étapes les plus proches de l'entrée ou de la sortie du U-Net, mais plutôt dans les couches intermédiaires et le \textit{bottleneck}, afin de capturer les dépendances à long terme sans augmenter démesurément la complexité du modèle.\\

L'architecture complète d'un block d'Attention pour un DDPM avec Classifier-Free Guidance est présentée en annexe \ref{annexe:cfg_resnet_architecture}, figure \ref{fig:attention_block}.\\

Ayant présenté les composants clés de l'architecture d'un DDPM, nous pouvons désormais présenter les algorithmes d'entraînement et d'inférence associés à ce modèle.

\subsection{Algorithmes : Entraînement et Inférence}

La figure \ref{fig:ddpm_algos_combined} présente les algorithmes d'entraînement et d'inférence d'un DDPM, tels que décrits dans l'article original de \cite{Ho2020}. La suite de notre étude se base sur ces algorithmes, toutefois modifiés pour intégrer l'aspect de conditionnement sur des classes d'images (ie. pour implémenter un modèle de diffusion conditionnel).\\

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\linewidth]{images/training-sampling-ddpm.png}
    \caption{Algorithmes d'entraînement et d'échantillonnage des DDPM. \cite{Ho2020}}
    \label{fig:ddpm_algos_combined} 
\end{figure}

Si les algorithmes présentés (\ref{fig:ddpm_algos_combined}) sont assez cohérents au regard des principes théoriques que nous avons exposés, il demeure un point d'intérêt que nous n'avons pas évoqué. En effet, lors de l'échantillonnage, le modèle de diffusion utilise une formule légèrement différente de celle présentée dans l'équation \ref{eq:mu_q_x_t-1_xt_epsilon} pour mettre à jour l'état $x_{t-1}$ à partir de $x_t$. En effet, la formule utilisée pour l'échantillonnage est la suivante :\\

\begin{equation}
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z
\label{eq:ddpm_sampling}
\end{equation}

avec $z \sim \mathcal{N}(0, \mathbf{I})$ si $t > 1$, et $z = 0$ sinon.\\

L'ajout du terme de bruit $\sigma_t z$ dans la formule d'échantillonnage permet d'introduire une certaine diversité dans les échantillons générés: en effet, sans ce terme de bruit, le processus de diffusion inverse serait entièrement déterministe, et le modèle générerait toujours la même image à partir du même échantillon initial $x_T$. En introduisant ce terme de bruit, nous nous plaçons dans un cadre de génération stochastique, où le modèle peut générer différentes images à partir du même échantillon initial $x_T$, en fonction du bruit aléatoire ajouté à chaque étape du processus de diffusion inverse. Bien entendu, nous n'ajoutons du bruit que pour les étapes intermédiaires du processus de diffusion inverse, et pas pour la dernière étape (lorsque $t=1$), afin de garantir que le modèle génère une image cohérente à la fin du processus.\\
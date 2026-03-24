\chapter{Algorithmes d'Entraînement et d'Échantillonage}
\label{annexe:algos}
\rhead{Algorithmes d'Entraînement et d'Échantillonage}

Cette annexe présente les algorithmes complets qui ont été mentionnés dans le rapport sans être détaillés.

\section*{Algorithme d'Entraînement du DDPM avec Classifier-Free Guidance}

\begin{algorithm}[H]
\DontPrintSemicolon
\SetAlgoLined
\textbf{Entrées :} Jeu de données $\mathcal{D}=\{(x_0,y)\}$, modèle de diffusion conditionnel $\epsilon_\theta$, table d'embeddings $E$, probabilité de dropout $p_{\text{uncond}}$\;\\
\textbf{Initialisation :}\\
Construire la suite de bruitage $\{\alpha_t\}_{t=1}^T$ et les produits cumulés $\{\bar{\alpha}_t\}_{t=1}^T$\;\\
Initialiser l'optimiseur sur les paramètres de $\epsilon_\theta$ et de $E$\;\\
\For{$epochs = 1$ \KwTo $num\_epochs$}{
    \For{chaque mini-batch $(x_0, y)$}{
    
        \textbf{1. Construction d'un exemple bruité :}\\
        Échantillonner un pas de temps $t \sim \mathcal{U}(\{1,\dots,T\})$\;\\
        Échantillonner un bruit gaussien $\varepsilon \sim \mathcal{N}(0,\mathbf{I})$\;\\
        Construire l'image bruitée :
        $x_t \leftarrow \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon$\;\\
    
        \textbf{2. Construction du conditionnement de classe :}\\
        Récupérer l'embedding de classe : $y_{\text{emb}} \leftarrow E[y]$\;\\
        Échantillonner $u \sim \mathcal{U}(0,1)$\;\\
        \If{$u < p_{\text{uncond}}$}{
            $y_{\text{emb}} \leftarrow \mathbf{0}$
        } 
        \vspace{0.25cm}
        \textbf{3. Prédiction du bruit et apprentissage :}\\
        Prédire le bruit : $\hat{\varepsilon} \leftarrow \epsilon_\theta(x_t,t,y_{\text{emb}})$\;\\
        Calculer la perte : $\mathcal{L} \leftarrow \|\hat{\varepsilon}-\varepsilon\|_2^2$\;\\
        Mettre à jour les paramètres de $\epsilon_\theta$ et de $E$ par descente de gradient\;
    }
}
\caption{Entraînement d'un DDPM avec \textit{Classifier-Free Guidance}}
\label{alg:cfg_training}
\end{algorithm}

\section*{Algorithme d'Échantillonage avec Classifier-Free Guidance}

\begin{algorithm}[H]
\DontPrintSemicolon
\SetAlgoLined
\textbf{Entrées :} Modèle de diffusion conditionnel $\epsilon_\theta$, table d'embeddings $E$, échelle de guidage $s$, classe cible $y$\;\\
\textbf{Initialisation :}\\
$x_T \sim \mathcal{N}(0, \mathbf{I})$\\
Construire la suite de bruitage $\{\alpha_t\}_{t=1}^T$, les produits cumulés $\{\bar{\alpha}_t\}_{t=1}^T$ et les écarts-types $\{\sigma_t\}_{t=1}^T$\;\\
Construire l'embedding conditionnel $y_{\text{cond}} \leftarrow E[y]$\;\\
Construire l'embedding inconditionnel $y_{\varnothing} \leftarrow \mathbf{[0, .., 0]}$\;\\
\For{$t = T$ \KwTo $1$}{
    \textbf{1. Double prédiction du modèle de diffusion :}\\
    Prédire le bruit conditionnel : $\epsilon_t^{\text{cond}} \leftarrow \epsilon_\theta(x_t,t,y_{\text{cond}})$\;\\
    Prédire le bruit inconditionnel : $\epsilon_t^{\text{uncond}} \leftarrow \epsilon_\theta(x_t,t,y_{\varnothing})$\;\\

    \textbf{2. Combinaison des prédictions (CFG) :}\\
    Combiner les deux estimations selon: 
    $\hat{\epsilon}_t \leftarrow (1+s)\,\epsilon_t^{\text{cond}} - s\,\epsilon_t^{\text{uncond}}$\;\\

    \textbf{3. Échantillonnage de $x_{t-1}$ :}\\
    Calculer la moyenne $\mu_t$ (selon la formule des DDPM) :
    $\mu_t \leftarrow \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\epsilon}_t \right)$\;\\
    $z \sim \mathcal{N}(0, \mathbf{I})$ si $t > 1$, sinon $z = 0$\;
    $x_{t-1} \leftarrow \mu_t + \sigma_t z$\;\\
}
\vspace{0.2cm}
\Return $x_0$
\caption{Échantillonnage avec \textit{Classifier-Free Guidance}}
\label{alg:cfg_inference}
\end{algorithm}

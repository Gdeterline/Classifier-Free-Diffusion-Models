\chapter{Classifier-Free Guidance - Architecture d'un Block de ResNet Conditionnel}
\label{annexe:cfg_resnet_architecture}
\rhead{Classifier-Free Guidance - Architecture d'un Block de ResNet Conditionnel}

La figure \ref{fig:resnet_block_cfg} illustre l'architecture d'un block de ResNet conditionnel, pour un DDPM avec Classifier-Free Guidance. Nous avons les mêmes composantes que pour un block de ResNet inconditionnel (convolutions, normalisation, activation), mais avec la notion de modulation des canaux pour intégrer les informations du pas de temps mais aussi de la classe.\\

\begin{figure}[htbp]
\centering
    \resizebox{!}{0.3\paperheight}{
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
        \node[draw, fill=gray!10, left=1.2cm of mult, font=\tiny, text width=2.5cm, align=center] (res_proj) {
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
        \draw[-{Stealth}] (in.west) -- ++(-1.8,0) -- (res_proj.north);
        \draw[-{Stealth}] (res_proj.south) |- (plus_res.west);
    \end{tikzpicture}
    }
    \caption{Architecture d'un block de ResNet conditionnel pour un DDPM avec Classifier-Free Guidance.}
    \label{fig:resnet_block_cfg}
\end{figure}

Notons plusieurs aspects importants concernant l'architecture du block de ResNet en figure \ref{fig:resnet_block_cfg} :
\begin{itemize}
    \item Nous avons une skip connection (chemin résiduel) qui permet de faire passer l'entrée $x$ directement à la sortie du block, ce qui stabilise l'entraînement.
    \item L'utilisation de GroupNorm permet de diviser les canaux en groupes pour la normalisation, et est particulièrement adaptée dans la mesure où nous travaillons sur les canaux de l'image.
    \item Le dropout présent dans la deuxième partie du block de ResNet permet de régulariser le modèle et n'a pas d'influence sur la classe des images générées.
\end{itemize}
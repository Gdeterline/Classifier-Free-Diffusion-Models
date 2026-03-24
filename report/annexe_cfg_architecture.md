\chapter{Architectures détaillées}
\label{annexe:cfg_resnet_architecture}
\rhead{Architectures détaillées}

\section{Architecture d'un Block de ResNet non conditionnel - DDPM}

La figure \ref{fig:resnet_block_ddpm} illustre l'architecture d'un block de ResNet non conditionnel, pour un DDPM. Nous avons les composantes classiques d'un block de ResNet (convolutions, normalisation, activation), mais avec la notion d'injection du pas de temps $t$ pour moduler les canaux de l'image, ce qui permet au même modèle de gérer les différentes étapes du processus de diffusion.\\

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

Notons plusieurs aspects importants concernant l'architecture du block de ResNet en figure \ref{fig:resnet_block_cfg} :
\begin{itemize}
    \item Nous avons une skip connection (chemin résiduel) qui permet de faire passer l'entrée $x$ directement à la sortie du block, ce qui stabilise l'entraînement.
    \item L'utilisation de GroupNorm permet de diviser les canaux en groupes pour la normalisation, et est particulièrement adaptée dans la mesure où nous travaillons sur les canaux de l'image.
    \item Le dropout présent dans la deuxième partie du block de ResNet permet de régulariser le modèle et n'a pas d'influence sur la classe des images générées.
\end{itemize}

\newpage

\section{Architecture d'un block d'Attention - DDPM}

La figure \ref{fig:attention_block} illustre l'architecture d'un block d'Attention pour un DDPM, qui est utilisé pour capturer les dépendances à long terme dans les images. Nous avons les composantes classiques d'un block d'Attention (Q, K, V, produit scalaire, softmax).

\begin{figure}[htbp]
\centering
    \resizebox{!}{0.3\paperheight}{
                \begin{tikzpicture}[
                    node distance=0.5cm,
                    every node/.style={font=\tiny, thick},
                    input/.style={draw, rectangle, fill=green!5, minimum width=1.5cm, minimum height=0.4cm},
                    block/.style={draw, rectangle, fill=blue!10, minimum width=3cm, minimum height=0.5cm, align=center},
                    op/.style={draw, circle, fill=orange!20, inner sep=1pt, minimum size=0.5cm},
                    annot/.style={font=\tiny\itshape, color=gray}
                ]
                    % --- ENTRÉES ---
                    \node[input] (k) {Key ($K$)};
                    \node[input, left=1cm of k] (q) {Query ($Q$)};
                    \node[input, right=1cm of k] (v) {Value ($V$)};
                    \node[annot, above=0.2cm of k] {Dimensions : $(L, d_k)$};
        
                    % --- CALCUL DU SCORE ---
                    \node[op, below=0.8cm of k] (dot1) {$\times$};
                    \node[right=0.02cm of dot1, font=\tiny\bfseries] {Produit Scalaire};
                    
                    \draw[->] (q) -- (dot1) node[pos=0.4, left] {$Q$};
                    \draw[->] (k) -- (dot1) node[pos=0.4, right] {$K$};
        
                    % --- NORMALISATION ---
                    \node[block, below=0.5cm of dot1] (scale) {Scaling ($1/\sqrt{d_k}$)};
                    \node[block, below=0.5cm of scale] (soft) {Softmax};
                    
                    \draw[->] (dot1) -- (scale);
                    \draw[->] (scale) -- (soft);
        
                    % --- APPLICATION SUR V ---
                    \node[op, below=1.2cm of soft] (dot2) {$\times$};
                    \node[right=0.2cm of dot2, yshift=-0.3cm, font=\tiny\bfseries] {Somme pondérée};        
                    
                    \draw[->] (soft) -- (dot2) node[pos=0.5, left] {Scores $\alpha$};
                    \draw[->] (v) |- (dot2) node[pos=0.7, above] {$V$};
        
                    % --- SORTIE ---
                    \node[input, below=0.7cm of dot2] (out) {Attention Out};
                    \draw[->] (dot2) -- (out);
                \end{tikzpicture}
            }
    \caption{Architecture d'un block d'Attention pour un DDPM.}
    \label{fig:attention_block}
\end{figure}

\newpage

\section{Architecture d'un Block de ResNet Conditionnel - Classifier-Free Guidance}

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
\chapter{Échantillonage sur le jeu de données CIFAR10}
\label{annexe:cfg_cifar10}
\rhead{Échantillonage sur le jeu de données CIFAR10}

Cette annexe présente les résultats de nos expérimentations avec la Classifier Guidance ainsi que la Classifier-Free Guidance (CFG) sur le dataset CIFAR10.

\section*{Échantillons du jeu de données CIFAR10}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{./images/CIFAR10_image_grid.png}
    \caption{Exemples d'images du dataset CIFAR10.}
    \label{fig:cifar10_samples}
\end{figure}

La figure \ref{fig:cifar10_samples} présente une grille d'images extraites du dataset CIFAR10, qui contient 60 000 images de 32x32 pixels réparties en 10 classes différentes (avion, automobile, oiseau, chat, cerf, chien, grenouille, cheval, bateau et camion).

\newpage

\section*{Résultats de la Classifier Guidance sur CIFAR10}
\label{annexe:cg_cifar10}
La figure \ref{fig:guidance_frog_comparison} compare la génération sans guidance (échantillonnage inconditionnel) et la génération guidée avec deux niveaux d'intensité.

\begin{figure}[H]
    \centering
    % --- GS = 0 ---
    \begin{subfigure}[b]{0.5\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/cifar10_ep500_grenouille_uncond_gs0.png}
        \caption{Guidance nulle ($GS=0$) : Le modèle génère des classes aléatoires.}
        \label{fig:frog_gs0}
    \end{subfigure}

    \vspace{0.8cm} % Espace entre les lignes

    % --- GS = 3 ---
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/cifar10_ep500_grenouille_guided_gs3.png}
        \caption{Guidance modérée ($GS=3$).}
        \label{fig:frog_gs3}
    \end{subfigure}
    \hfill
    % --- GS = 10 ---
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/cifar10_ep500_grenouille_guided_gs10.png}
        \caption{Guidance forte ($GS=10$).}
        \label{fig:frog_gs10}
    \end{subfigure}

    \caption{Évolution de la génération pour la classe "grenouille".}
    \label{fig:guidance_frog_comparison}
\end{figure}

\newpage

\section*{Résultats de la Classifier-Free Guidance sur CIFAR10}

La figure \ref{fig:cfg_results_cifar10} présente les échantillons générés par notre modèle de diffusion guidée sans classifieur (CFG) sur le dataset CIFAR10, pour différentes valeurs du paramètre de guidance $s$.

\begin{figure}[ht!]
    \centering
    \small % Réduit légèrement la taille des polices dans la figure
    
    % --- Première ligne (3 images) ---
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/guided_unet_frog_s-1.0.png}
        \caption{\scriptsize $s=-1$}
        \label{fig:cfg_uncond}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/guided_unet_frog_s-0.5.png}
        \caption{\scriptsize $s=-0.5$}
        \label{fig:cfg_uncond_cond}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/guided_unet_frog_s0.0.png}
        \caption{\scriptsize $s=0$}
        \label{fig:cfg_cond0}
    \end{subfigure}

    \vspace{0.2cm} % Espace minimal entre les deux lignes

    % --- Deuxième ligne (2 images centrées) ---
    \begin{subfigure}[b]{0.38\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/guided_unet_frog_s3.0.png}
        \caption{\scriptsize $s=3$}
        \label{fig:cfg_cond3}
    \end{subfigure}
    \hspace{0.5cm}
    \begin{subfigure}[b]{0.38\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/guided_unet_frog_s10.0.png}
        \caption{\scriptsize $s=10$}
        \label{fig:cfg_cond10}
    \end{subfigure}

    \vspace{0.1cm}
    \caption{Échantillons DDPM avec CFG (classe grenouille) selon le facteur d'échelle $s$.}
    \label{fig:cfg_results_cifar10}
\end{figure}

La figure \ref{fig:cfg_cond3} montre que le modèle parvient à générer des images reconnaissables à première vue, mais qui présentent des artefacts et une qualité globale inférieure à celle obtenue sur MNIST. Si nous ne pouvons nous attendre à des images générées de très haute qualité (les images du jeu de données restent de taille 32x32, et donc de qualité limitée, figure \ref{fig:cifar10_samples}), nous pouvons néanmoins observer que le modèle est capable de générer des images qui sont reconnaissables comme appartenant à la classe cible, ce qui montre que la Classifier-Free Guidance fonctionne également sur ce jeu de données plus complexe. Il serait cependant pertinent de considérer une architecture plus complexe (par exemple, en augmentant la profondeur du modèle) pour améliorer la qualité des images générées sur ce jeu de données (ou d'autres, du type ImageNet).\\

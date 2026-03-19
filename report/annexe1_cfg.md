\chapter{Classifier-Free Guidance - CIFAR10}
\label{annexe:cfg_cifar10}
\rhead{Classifier-Free Guidance - CIFAR10}

Cette annexe présente les résultats de nos expérimentations avec la Classifier-Free Guidance (CFG) sur le dataset CIFAR10.

\section*{Échantillons du jeu de données CIFAR10}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{./images/CIFAR10_image_grid.png}
    \caption{Exemples d'images du dataset CIFAR10.}
    \label{fig:cifar10_samples}
\end{figure}

La figure \ref{fig:cifar10_samples} présente une grille d'images extraites du dataset CIFAR10, qui contient 60 000 images de 32x32 pixels réparties en 10 classes différentes (avion, automobile, oiseau, chat, cerf, chien, grenouille, cheval, bateau et camion).

\section*{Résultats de la Classifier-Free Guidance sur CIFAR10}

La figure \ref{fig:cfg_results_cifar10} présente les échantillons générés par notre modèle de diffusion guidée sans classifieur (CFG) sur le dataset CIFAR10, pour différentes valeurs du paramètre de guidance $s$.

\begin{figure}[htbp]
    \centering
    % --- Première ligne (2 images) ---
    \begin{subfigure}[b]{0.42\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/guided_unet_frog_s-1.0.png}
        \caption{Classe $\emptyset$ | $s=-1$}
        \label{fig:cfg_uncond}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.42\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/guided_unet_frog_s-0.5.png}
        \caption{Classe grenouille | $s=-0.5$}
        \label{fig:cfg_uncond_cond}
    \end{subfigure}

    % --- Deuxième ligne (2 images) ---
    \begin{subfigure}[b]{0.42\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/guided_unet_frog_s0.0.png}
        \caption{Classe grenouille | $s=0$}
        \label{fig:cfg_cond0}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.42\textwidth}
        \centering
        \includegraphics[width=\textwidth]{images/guided_unet_frog_s3.0.png} % <-- Votre nouvelle image pour s=3
        \caption{Classe grenouille | $s=3$}
        \label{fig:cfg_cond3}
    \end{subfigure}

    % --- Troisième ligne (1 image centrée) ---
    \begin{subfigure}[b]{0.42\textwidth} % Même largeur pour la cohérence
        \centering
        \includegraphics[width=\textwidth]{images/guided_unet_frog_s10.0.png}
        \caption{Classe grenouille | $s=10$}
        \label{fig:cfg_cond10}
    \end{subfigure}

    \caption{Exemples d'images générées par un DDPM avec Classifier-Free Guidance (classe grenouille) pour différentes valeurs de guidance $s$.}
    \label{fig:cfg_results_cifar10}
\end{figure}
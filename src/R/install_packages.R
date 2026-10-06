# ==============================================================================
# Script de Instalación Automatizada de Paquetes para la Actividad 2
# Maestría en Ciencia de Datos - Universidad Santo Tomás
# ==============================================================================

cat(">>> Verificando e instalando paquetes de R requeridos...\n")

paquetes <- c(
  "readxl",       # Lectura de archivos Excel (.xlsx)
  "dplyr",        # Manipulación y transformación de datos
  "tidyr",        # Organización de datos
  "ggplot2",      # Gráficos estadísticos de alta calidad
  "cluster",      # Algoritmos de agrupamiento (daisy, pam, silhouette)
  "factoextra",   # Visualización factorial y de clústeres
  "FactoMineR",   # Análisis de Componentes Principales
  "fpc",          # Estadísticas de clústeres y clusterboot
  "clValid",      # Medidas de conectividad y estabilidad
  "MASS",         # Métodos estadísticos multivariados
  "dendextend",   # Manipulación de dendrogramas
  "mclust",       # Métricas de concordancia (Adjusted Rand Index)
  "knitr",        # Formateo de tablas
  "rmarkdown"     # Generación de informes reproducibles
)

# Paquetes faltantes
faltantes <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]

if (length(faltantes) > 0) {
  cat("Instalando:", paste(faltantes, collapse = ", "), "\n")
  install.packages(faltantes, repos = "https://cloud.r-project.org", dependencies = TRUE)
} else {
  cat("✓ Todos los paquetes principales ya están instalados.\n")
}

# Paquetes complementarios para Davies-Bouldin
for (pkg in c("clusterSim", "clusterCrit")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("Intentando instalar", pkg, "...\n")
    try(install.packages(pkg, repos = "https://cloud.r-project.org", dependencies = TRUE))
  }
}

cat("\n>>> ¡Instalación completada con éxito! El entorno está listo.\n")

# Validación del número de grupos en agrupamiento jerárquico
#
# vhc()
#
# D   : objeto dist o matriz de disimilitudes ya calculada.
# k   : números de grupos a evaluar, por ejemplo 1:8.
# m   : medidas:
#       "sil"  = Silhouette promedio
#       "dunn" = Dunn
#       "ch"   = Calinski-Harabasz
#       "db"   = Davies-Bouldin
#       "con"  = Connectivity
#       "gap"  = Gap statistic
#       "apn"  = Average Proportion of Non-overlap
#       "fom"  = Figure of Merit
#       "ps"   = Prediction Strength
#       "jac"  = Bootstrap Jaccard stability
# met : método de enlace de hclust.
#       "ward" se interpreta como "ward.D2".
# X   : matriz numérica usada para construir D.
#       Solo se requiere para db, gap, apn y fom.
# B   : número de remuestras para Gap y Jaccard.
# M   : número de particiones para Prediction Strength.
# nn  : número de vecinos para Connectivity.
# dp  : potencia de las distancias en Gap.
#       dp = 1 reproduce el valor histórico de clusGap;
#       dp = 2 corresponde a la formulación de Tibshirani et al.
# seed: semilla.
#
# Nota:
# D se usa directamente para la jerarquía principal y para todas las
# medidas que admiten una matriz de disimilitudes.
#
# Gap necesita X para generar muestras de referencia. Las distancias de
# esas muestras sí deben calcularse internamente, pues no existen en D.
#
# APN y FOM eliminan una variable cada vez; por ello deben construir las
# distancias de los datos reducidos. D no se recalcula.
#
# Prediction Strength trabaja directamente sobre D mediante
# fpc::disthclustCBI.

vhc = function(D, k, m,
               met = "ward",
               X = NULL,
               B = 100,
               M = 100,
               nn = 10,
               dp = 1,
               seed = 123){

  # -----------------------------
  # Comprobaciones
  # -----------------------------

  if (is.matrix(D)) D = stats::as.dist(D)

  if (!inherits(D, "dist"))
    stop("D debe ser un objeto 'dist' o una matriz de disimilitudes.")

  if (any(!is.finite(D)))
    stop("D contiene valores no finitos.")

  n = attr(D, "Size")

  k = sort(unique(as.integer(k)))

  if (length(k) == 0 || any(k < 1) || any(k >= n))
    stop("Los valores de k deben cumplir 1 <= k < n.")

  ok = c("sil", "dunn", "ch", "db", "con",
         "gap", "apn", "fom", "ps", "jac")

  m = unique(tolower(m))

  if (length(m) == 0 || any(!m %in% ok))
    stop(
      paste0(
        "m debe contener alguna de estas medidas: ",
        paste(ok, collapse = ", "), "."
      )
    )

  if (identical(met, "ward")) met = "ward.D2"

  met = match.arg(
    met,
    c("ward.D2", "ward.D", "complete", "single",
      "average", "mcquitty", "median", "centroid")
  )

  reqX = any(c("db", "gap", "apn", "fom") %in% m)

  if (reqX){

    if (is.null(X))
      stop("X es necesaria para db, gap, apn o fom.")

    X = as.matrix(X)

    if (!is.numeric(X))
      stop("X debe ser una matriz numérica.")

    if (nrow(X) != n)
      stop("X y D deben contener los mismos individuos y en el mismo orden.")

    if (any(!is.finite(X)))
      stop("X contiene valores no finitos.")
  }

  pk = character(0)

  if (any(c("sil", "dunn", "ch", "ps", "jac") %in% m))
    pk = c(pk, "fpc")

  if (any(c("con", "apn", "fom") %in% m))
    pk = c(pk, "clValid")

  if ("db" %in% m)
    pk = c(pk, "clusterSim")

  pk = unique(pk)

  faltan = pk[
    !vapply(pk, requireNamespace, logical(1), quietly = TRUE)
  ]

  if (length(faltan) > 0)
    stop(
      paste0(
        "Instale primero: ",
        paste(faltan, collapse = ", "),
        "."
      )
    )

  # -----------------------------
  # Jerarquía principal
  # -----------------------------

  h = stats::hclust(D, method = met)

  g = lapply(
    k,
    function(ki){
      if (ki == 1) rep(1L, n) else stats::cutree(h, k = ki)
    }
  )

  names(g) = k

  z = data.frame(k = k)

  # -----------------------------
  # Silhouette, Dunn y CH
  # -----------------------------

  if (any(c("sil", "dunn", "ch") %in% m)){

    a = t(vapply(
      seq_along(k),
      function(i){

        if (k[i] == 1)
          return(c(
            sil = NA_real_,
            dunn = NA_real_,
            ch = NA_real_
          ))

        s = fpc::cluster.stats(
          D,
          clustering = g[[i]],
          silhouette = "sil" %in% m,
          G2 = FALSE,
          G3 = FALSE,
          wgap = FALSE,
          sepindex = FALSE,
          aggregateonly = TRUE
        )

        c(
          sil = if ("sil" %in% m) s$avg.silwidth else NA_real_,
          dunn = if ("dunn" %in% m) s$dunn else NA_real_,
          ch = if ("ch" %in% m) s$ch else NA_real_
        )
      },
      numeric(3)
    ))

    if ("sil" %in% m)  z$sil = a[, "sil"]
    if ("dunn" %in% m) z$dunn = a[, "dunn"]
    if ("ch" %in% m)   z$ch = a[, "ch"]
  }

  # -----------------------------
  # Davies-Bouldin
  # -----------------------------
  # Se usa la versión basada en medoides de clusterSim,
  # que utiliza directamente la matriz de distancias D.

  if ("db" %in% m){

    z$db = vapply(
      seq_along(k),
      function(i){

        if (k[i] == 1) return(NA_real_)

        clusterSim::index.DB(
          x = X,
          cl = g[[i]],
          d = D,
          centrotypes = "medoids"
        )$DB
      },
      numeric(1)
    )
  }

  # -----------------------------
  # Connectivity
  # -----------------------------

  if ("con" %in% m){

    A = as.matrix(D)
    nn1 = min(as.integer(nn), n - 1L)

    z$con = vapply(
      seq_along(k),
      function(i)
        clValid::connectivity(
          distance = A,
          clusters = g[[i]],
          neighbSize = nn1
        ),
      numeric(1)
    )
  }

  # -----------------------------
  # Gap statistic
  # -----------------------------
  # La dispersión observada usa directamente D.
  # Las muestras de referencia se generan como en clusGap
  # (uniformes en el hiperrectángulo después de rotación PCA).
  # En las muestras de referencia se usa distancia euclídea.

  if ("gap" %in% m){

    W = function(d, cl){

      A = as.matrix(d)^dp
      sp = split(seq_along(cl), cl)

      sum(vapply(
        sp,
        function(ii){

          ni = length(ii)

          if (ni <= 1) return(0)

          0.5 * sum(A[ii, ii][upper.tri(A[ii, ii])]) / ni
        },
        numeric(1)
      ))
    }

    wo = vapply(
      seq_along(k),
      function(i) W(D, g[[i]]),
      numeric(1)
    )

    if (any(wo <= 0))
      stop("Gap no puede calcularse porque alguna dispersión observada es cero.")

    xs = scale(X, center = TRUE, scale = FALSE)
    cen = attr(xs, "scaled:center")

    V = svd(xs, nu = 0)$v
    xp = xs %*% V

    rg = apply(xp, 2, range)

    set.seed(seed)

    lr = matrix(
      NA_real_,
      nrow = B,
      ncol = length(k)
    )

    for (b in seq_len(B)){

      u = vapply(
        seq_len(ncol(xp)),
        function(j)
          stats::runif(
            n,
            min = rg[1, j],
            max = rg[2, j]
          ),
        numeric(n)
      )

      u = matrix(u, nrow = n)

      xb = u %*% t(V)
      xb = sweep(xb, 2, cen, "+")

      db = stats::dist(xb)
      hb = stats::hclust(db, method = met)

      gb = lapply(
        k,
        function(ki){
          if (ki == 1) rep(1L, n) else stats::cutree(hb, k = ki)
        }
      )

      lr[b, ] = log(
        vapply(
          seq_along(k),
          function(i) W(db, gb[[i]]),
          numeric(1)
        )
      )
    }

    z$gap = colMeans(lr) - log(wo)
  }

  # -----------------------------
  # APN y FOM
  # -----------------------------
  # Para cada variable eliminada se necesita una nueva jerarquía.
  # Se usa distancia euclídea en los datos reducidos.
  # D permanece sin recalcular.

  if (any(c("apn", "fom") %in% m)){

    p = ncol(X)

    if (p < 2)
      stop("APN y FOM requieren al menos dos variables en X.")

    hd = lapply(
      seq_len(p),
      function(j){

        Xj = X[, -j, drop = FALSE]

        stats::hclust(
          stats::dist(Xj),
          method = met
        )
      }
    )

    sf = t(vapply(
      seq_along(k),
      function(i){

        st = vapply(
          seq_len(p),
          function(j){

            gj = if (k[i] == 1)
              rep(1L, n)
            else
              stats::cutree(hd[[j]], k = k[i])

            clValid::stability(
              mat = X,
              Dist = D,
              del = j,
              cluster = g[[i]],
              clusterDel = gj
            )
          },
          numeric(4)
        )

        rowMeans(st)
      },
      numeric(4)
    ))

    colnames(sf) = c("APN", "AD", "ADM", "FOM")

    if ("apn" %in% m) z$apn = sf[, "APN"]
    if ("fom" %in% m) z$fom = sf[, "FOM"]
  }

  # -----------------------------
  # Prediction Strength
  # -----------------------------
  # Se trabaja directamente con D.
  # La regla de clasificación se adapta cuando existe una
  # correspondencia natural con el método jerárquico.

  if ("ps" %in% m){

    cl = switch(
      met,
      "single" = "knn",
      "complete" = "fn",
      "average" = "averagedist",
      "averagedist"
    )

    pv = rep(NA_real_, length(k))
    pv[k == 1] = 1

    if (any(k >= 2)){

      set.seed(seed)

      pr = fpc::prediction.strength(
        D,
        Gmin = 2,
        Gmax = max(k),
        M = M,
        cutoff = 0,
        clustermethod = fpc::disthclustCBI,
        classification = cl,
        nnk = 1,
        distances = TRUE,
        method = met,
        cut = "number",
        count = FALSE
      )

      ii = which(k >= 2)
      pv[ii] = pr$mean.pred[k[ii]]
    }

    z$ps = pv
  }

  # -----------------------------
  # Bootstrap Jaccard stability
  # -----------------------------

  if ("jac" %in% m){

    z$jac = vapply(
      seq_along(k),
      function(i){

        if (k[i] == 1) return(1)

        jb = fpc::clusterboot(
          D,
          B = B,
          bootmethod = "boot",
          clustermethod = fpc::disthclustCBI,
          k = k[i],
          cut = "number",
          method = met,
          seed = seed,
          count = FALSE
        )

        mean(jb$bootmean)
      },
      numeric(1)
    )
  }

  z
}

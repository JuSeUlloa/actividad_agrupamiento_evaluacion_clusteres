# Validación del número de grupos para k-means y k-medoides (PAM)
#
# vkm()
#
# X   : matriz numérica de datos. Para k-means es obligatoria.
#       Para PAM puede usarse X o una matriz de disimilitudes D.
# D   : objeto dist o matriz de disimilitudes precomputada.
#       Si se proporciona D para PAM puede usarse X = NULL.
# k   : números de grupos que se evaluarán.
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
# alg : "kmeans" o "pam"
# B   : número de remuestras para Gap y Jaccard.
# M   : número de particiones para Prediction Strength.
# nn  : número de vecinos para Connectivity.
# ns  : número de inicios de k-means.
# seed: semilla.
#
# La función calcula únicamente las medidas incluidas en m.
# No decide si una medida es apropiada para los datos suministrados.
#
# Si X se suministra sin D, la distancia euclídea se calcula una sola vez
# cuando alguna medida la necesita.
#
# Si D se suministra para PAM, D se usa directamente y no se recalcula.

vkm = function(X = NULL, D = NULL, k, m,
              alg = c("kmeans", "pam"),
              B = 100, M = 100, nn = 10,
              ns = 25, seed = 123){

  alg = match.arg(alg)

  ok = c("sil", "dunn", "ch", "db", "con",
         "gap", "apn", "fom", "ps", "jac")

  m = unique(tolower(m))
  k = sort(unique(as.integer(k)))

  if (length(m) == 0 || any(!m %in% ok))
    stop(
      paste0(
        "m debe contener alguna de estas medidas: ",
        paste(ok, collapse = ", "), "."
      )
    )

  if (length(k) == 0 || any(k < 1))
    stop("k debe contener enteros positivos.")

  if (alg == "kmeans" && is.null(X))
    stop("k-means requiere X.")

  if (alg == "pam" && is.null(X) && is.null(D))
    stop("PAM requiere X o D.")

  if (!is.null(X)){

    X = as.matrix(X)

    if (!is.numeric(X))
      stop("X debe ser numérica.")

    if (any(!is.finite(X)))
      stop("X contiene valores no finitos.")

    n = nrow(X)

  } else {

    if (is.matrix(D)) D = stats::as.dist(D)

    if (!inherits(D, "dist"))
      stop("D debe ser un objeto 'dist' o una matriz de disimilitudes.")

    if (any(!is.finite(D)))
      stop("D contiene valores no finitos.")

    n = attr(D, "Size")
  }

  if (!is.null(D)){

    if (is.matrix(D)) D = stats::as.dist(D)

    if (!inherits(D, "dist"))
      stop("D debe ser un objeto 'dist' o una matriz de disimilitudes.")

    if (any(!is.finite(D)))
      stop("D contiene valores no finitos.")

    if (!is.null(X) && attr(D, "Size") != nrow(X))
      stop("X y D deben contener los mismos individuos.")
  }

  if (any(k >= n))
    stop("Los valores de k deben ser menores que el número de individuos.")

  # -----------------------------
  # Paquetes
  # -----------------------------

  pk = character(0)

  if (alg == "pam")
    pk = c(pk, "cluster")

  if (any(c("sil", "dunn", "ch", "ps", "jac") %in% m))
    pk = c(pk, "fpc")

  if (any(c("con", "apn", "fom") %in% m))
    pk = c(pk, "clValid")

  if ("db" %in% m && alg == "kmeans")
    pk = c(pk, "clusterCrit")

  if ("gap" %in% m)
    pk = c(pk, "cluster")

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
  # Funciones auxiliares
  # -----------------------------

  fitX = function(x, kk, ss = seed){

    if (kk == 1)
      return(rep(1L, nrow(x)))

    if (alg == "kmeans"){

      set.seed(ss)

      return(
        stats::kmeans(
          x,
          centers = kk,
          nstart = ns
        )$cluster
      )
    }

    as.integer(
      cluster::pam(
        x,
        k = kk,
        diss = FALSE,
        cluster.only = TRUE
      )
    )
  }

  fitD = function(d, kk){

    if (kk == 1)
      return(rep(1L, attr(d, "Size")))

    if (alg != "pam")
      stop("D solo puede usarse directamente para PAM.")

    as.integer(
      cluster::pam(
        d,
        k = kk,
        diss = TRUE,
        cluster.only = TRUE
      )
    )
  }

  # Davies-Bouldin basado en medoides y D
  dbmed = function(d, cl){

    A = as.matrix(d)
    lev = sort(unique(cl))
    q = length(lev)

    med = integer(q)
    s = numeric(q)

    for (i in seq_along(lev)){

      ii = which(cl == lev[i])
      Ai = A[ii, ii, drop = FALSE]

      med[i] = ii[which.min(rowSums(Ai))]
      s[i] = mean(A[ii, med[i]])
    }

    R = matrix(-Inf, q, q)

    for (i in seq_len(q)){
      for (j in seq_len(q)){
        if (i != j)
          R[i, j] = (s[i] + s[j]) / A[med[i], med[j]]
      }
    }

    mean(apply(R, 1, max))
  }

  # -----------------------------
  # Distancia para X
  # -----------------------------

  needD = any(c("sil", "dunn", "ch", "db", "con") %in% m)

  if (is.null(D) && !is.null(X) && needD)
    D = stats::dist(X)

  # -----------------------------
  # Particiones principales
  # -----------------------------

  if (!is.null(X)){

    g = lapply(
      seq_along(k),
      function(i) fitX(X, k[i], seed + i - 1L)
    )

  } else {

    g = lapply(
      k,
      function(kk) fitD(D, kk)
    )
  }

  names(g) = k

  z = data.frame(k = k)

  # -----------------------------
  # Silhouette, Dunn y CH
  # -----------------------------

  if (any(c("sil", "dunn", "ch") %in% m)){

    if (is.null(D))
      D = stats::dist(X)

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

  if ("db" %in% m){

    if (alg == "kmeans"){

      if (is.null(X))
        stop("Davies-Bouldin para k-means requiere X.")

      z$db = vapply(
        seq_along(k),
        function(i){

          if (k[i] == 1) return(NA_real_)

          as.numeric(
            clusterCrit::intCriteria(
              X,
              part = as.integer(g[[i]]),
              crit = "Davies_Bouldin"
            )[[1]]
          )
        },
        numeric(1)
      )

    } else {

      if (is.null(D))
        D = stats::dist(X)

      z$db = vapply(
        seq_along(k),
        function(i){

          if (k[i] == 1) return(NA_real_)

          dbmed(D, g[[i]])
        },
        numeric(1)
      )
    }
  }

  # -----------------------------
  # Connectivity
  # -----------------------------

  if ("con" %in% m){

    if (is.null(D))
      D = stats::dist(X)

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

  if ("gap" %in% m){

    if (is.null(X))
      stop("Gap requiere X.")

    if (alg == "kmeans"){

      fg = function(x, kk){

        set.seed(seed + kk)

        list(
          cluster = stats::kmeans(
            x,
            centers = kk,
            nstart = ns
          )$cluster
        )
      }

    } else {

      fg = function(x, kk)
        list(
          cluster = cluster::pam(
            x,
            k = kk,
            diss = FALSE
          )$clustering
        )
    }

    set.seed(seed)

    ga = cluster::clusGap(
      X,
      FUNcluster = fg,
      K.max = max(k),
      B = B,
      verbose = FALSE
    )

    z$gap = ga$Tab[k, "gap"]
  }

  # -----------------------------
  # APN y FOM
  # -----------------------------

  if (any(c("apn", "fom") %in% m)){

    if (is.null(X))
      stop("APN y FOM requieren X.")

    p = ncol(X)

    if (p < 2)
      stop("APN y FOM requieren al menos dos variables.")

    if (is.null(D))
      D = stats::dist(X)

    sf = t(vapply(
      seq_along(k),
      function(i){

        if (k[i] == 1)
          return(c(APN = 0, AD = 0, ADM = 0, FOM = 1))

        st = vapply(
          seq_len(p),
          function(j){

            Xj = X[, -j, drop = FALSE]
            gj = fitX(
              Xj,
              k[i],
              seed + 10000L + 100L * i + j
            )

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

  if ("ps" %in% m){

    pv = rep(NA_real_, length(k))
    pv[k == 1] = 1

    if (any(k >= 2)){

      set.seed(seed)

      if (alg == "kmeans"){

        if (is.null(X))
          stop("Prediction Strength para k-means requiere X.")

        pr = suppressWarnings(
          fpc::prediction.strength(
            X,
            Gmin = 2,
            Gmax = max(k),
            M = M,
            clustermethod = fpc::kmeansCBI,
            classification = "centroid",
            cutoff = 0,
            scaling = FALSE,
            runs = ns,
            count = FALSE
          )
        )

      } else if (!is.null(D) && is.null(X)){

        pr = suppressWarnings(
          fpc::prediction.strength(
            D,
            Gmin = 2,
            Gmax = max(k),
            M = M,
            clustermethod = fpc::claraCBI,
            classification = "centroid",
            cutoff = 0,
            distances = TRUE,
            usepam = TRUE,
            diss = TRUE,
            count = FALSE
          )
        )

      } else {

        pr = suppressWarnings(
          fpc::prediction.strength(
            X,
            Gmin = 2,
            Gmax = max(k),
            M = M,
            clustermethod = fpc::claraCBI,
            classification = "centroid",
            cutoff = 0,
            distances = FALSE,
            usepam = TRUE,
            diss = FALSE,
            count = FALSE
          )
        )
      }

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

        if (alg == "kmeans"){

          jb = fpc::clusterboot(
            X,
            B = B,
            bootmethod = "boot",
            clustermethod = fpc::kmeansCBI,
            k = k[i],
            runs = ns,
            scaling = FALSE,
            seed = seed,
            count = FALSE
          )

        } else if (!is.null(D) && is.null(X)){

          jb = fpc::clusterboot(
            D,
            B = B,
            bootmethod = "boot",
            clustermethod = fpc::claraCBI,
            k = k[i],
            usepam = TRUE,
            diss = TRUE,
            seed = seed,
            count = FALSE
          )

        } else {

          jb = fpc::clusterboot(
            X,
            B = B,
            bootmethod = "boot",
            clustermethod = fpc::claraCBI,
            k = k[i],
            usepam = TRUE,
            diss = FALSE,
            seed = seed,
            count = FALSE
          )
        }

        mean(jb$bootmean)
      },
      numeric(1)
    )
  }

  z
}

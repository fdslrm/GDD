# Original C file for DE quadrature: Takuya Ooura, https://www.kurims.kyoto-u.ac.jp/~ooura/index.html
# R conversion and modification: Andrej Gajdos, https://github.com/fdslrm
# our application for numerical inversion of characteristic function

intde <- function(f, a, b, eps, ...) {
  
  # adjustable parameter 
  mmax <- 256
  efs <- 0.1
  hoff <- 8.5

  pi2 <- 2 * atan(1)
  epsln <- 1 - log(efs * eps)
  epsh <- sqrt(efs * eps)
  h0 <- hoff / epsln
  ehp <- exp(h0)
  ehm <- 1 / ehp 
  epst <- exp(-ehm * epsln)
  ba <- b - a
  ir <- f((a+b)*0.5) * (ba * 0.25)
  i <- ir * (2 * pi2)
  err <- abs(i) * epst
  h <- 2 * h0
  m <- 1
  
  repeat{
    iback <- i
    irback <- ir
    t <- h * 0.5
    
    repeat{
      em <- exp(t)
      ep <- pi2 * em
      em <- pi2 / em
      
      repeat{
        xw <- 1 / (1 + exp(ep - em))
        xa <- ba * xw
        wg <- xa * (1 - xw)
        fa <- f(a+xa) * wg
        fb <- f(b-xa) * wg
        ir <- ir + fa + fb
        i <- i + (fa + fb) * (ep + em)
        errt <- (abs(fa) + abs(fb)) * (ep + em)
        if (m == 1) {
          err <- err + errt * epst
        }
        ep <- ep * ehp
        em <- em * ehm
        
        if(!(errt > err || xw > epsh)){
          break
        }
      }
      
      t <- t + h
      
      if(!(t < h0)){
        break
      }
    }
    
    if (m == 1) {
      errh <- (err / epst) * epsh * h0
      errd <- 1 + 2 * errh
    } else {
      errd <- h * (abs(i - 2 * iback) + 4 * abs(ir - 2 * irback))
    }
    h <- h * 0.5
    m <- m * 2
    
    if(!(errd > errh && m < mmax)){
      break
    }
  }
  
  i = i * h
  
  if (errd > errh) {
    err = -errd * m
  } else {
    err = errh * epsh * m / (2 * efs)
  }
  
  result <- list("i" = i, "err" = err)
  
  return(result)
}

# example 

f1 <- function(x) {
  return(1 / sqrt(x))
}

f2 <- function(x) {
  return(sqrt(4 - x * x))
}

result1 <- intde(f1, 0, 10, 1e-15)
result1$i
result1$err

result2 <- intde(f2, 0, 2, 1e-15)
result2$i
result2$err


intdei <- function(f, a, eps, ...) {
  
  # adjustable parameter
  mmax <- 256
  efs <- 0.1
  hoff <- 11
  
  pi4 <- atan(1)
  epsln <- 1 - log(efs * eps)
  epsh <- sqrt(efs * eps)
  h0 <- hoff / epsln
  ehp <- exp(h0)
  ehm <- 1 / ehp
  epst <- exp(-ehm * epsln)
  ir <- f(a+1)
  i <- ir * (2 * pi4)
  err <- abs(i) * epst
  h <- 2 * h0
  m <- 1
  
  repeat {
    iback <- i
    irback <- ir
    t <- h * 0.5
    
    repeat {
      em <- exp(t)
      ep <- pi4 * em
      em <- pi4 / em
      
      repeat {
        xp <- exp(ep - em)
        xm <- 1 / xp
        fp <- f(a + xp) * xp
        fm <- f(a + xm) * xm
        ir <- ir + fp + fm
        i <- i + (fp + fm) * (ep + em)
        errt <- (abs(fp) + abs(fm)) * (ep + em)
        
        if (m == 1) {
          err <- err + errt * epst
        }
        
        ep <- ep * ehp
        em <- em * ehm
        
      if(!(errt > err || xm > epsh)) {
        break
      }
    }
      
      t <- t + h
      
    if(!(t < h0)) {
      break
    }
  }
    
    if (m == 1) {
      errh <- (err / epst) * epsh * h0
      errd <- 1 + 2 * errh
    } else {
      errd <- h * (abs(i - 2 * iback) + 4 * abs(ir - 2 * irback))
    }
    
    h <- h * 0.5
    m <- m * 2
    
  if(!(errd > errh && m < mmax)) {
    break
  }
}
  
  i <- i * h
  
  if (errd > errh) {
    err <- -errd * m
  } else {
    err <- errh * epsh * m / (2 * efs)
  }
  
  result <- list("i" = i, "err" = err)
  
  return(result)
}


f3 <- function(x) {
  return(1 / (1 + x * x))
}

f4 <- function(x) {
  return(exp(-x) / sqrt(x))
}
  

result3 <- intdei(f3, 0, 1e-15)
result3$i
result3$err

result4 <- intdei(f4, 0, 1e-15)
result4$i
result4$err


intdeo <- function(f, a, omega, eps, ...) {
  
  # adjustable parameter
  mmax <- 256
  lmax <- 5
  efs <- 0.1
  enoff <- 0.4
  pqoff <- 2.9
  ppoff <- -0.72
  
  pi4 <- atan(1)
  epsln <- 1 - log(efs * eps)
  epsh <- sqrt(efs * eps)
  n <- as.integer(enoff * epsln)
  frq4 <- abs(omega) / (2 * pi4)
  per2 <- 4 * pi4 / abs(omega)
  pq <- pqoff / epsln
  pp <- ppoff - log(pq * pq * frq4)
  ehp <- exp(2 * pq)
  ehm <- 1 / ehp
  xw <- exp(pp - 2 * pi4)
  i <- f(a + sqrt(xw * (per2 * 0.5)),...)
  ir <- i * xw
  i <- i * per2 * 0.5
  err <- abs(i)
  h <- 2
  m <- 1
  
  repeat {
    iback <- i
    irback <- ir
    t <- h * 0.5
    
    repeat {
      em = exp(2 * pq * t);
      ep = pi4 * em;
      em = pi4 / em;
      tk = t;
      
      repeat {
        xw <- exp(pp - ep - em)
        wg <- sqrt(frq4 * xw + tk * tk)
        xa <- xw / (tk + wg)
        wg <- (pq * xw * (ep - em) + xa) / wg
        fm <- f(a + xa,...)
        fp <- f(a + xa + per2 * tk,...)
        ir <- ir + (fp + fm) * xw
        fm <- fm * wg
        fp <- fp * (per2 - wg)
        i <- i + fp + fm
        if (m == 1) {
          err <- err +  abs(fp) + abs(fm)
        }
        ep <- ep * ehp
        em <- em * ehm
        tk <- tk + 1
        
      if(!(ep < epsln)) {
        break
      }
    }
      
      if (m == 1) {
        errh <- err * epsh
        err <- err * eps
      }
      
      tn <- tk
      
      while (abs(fm) > err) {
        xw <- exp(pp - ep - em);
        xa <- xw / tk * 0.5;
        wg <- xa * (1 / tk + 2 * pq * (ep - em));
        fm <- f(a + xa,...)
        ir <- ir + fm * xw
        fm <- fm * wg
        i <- i + fm
        ep <- ep * ehp
        em <- em * ehm
        tk <- tk + 1
      }
      
      fm <- f(a + per2 * tn,...)
      em <- per2 * fm
      i <- i + em
      
      if (abs(fp) > err || abs(em) > err) {
        
        l <- 0
        
        repeat {
          l <- l + 1
          tn <- tn + n
          em <- fm
          fm <- f(a + per2 * tn,...)
          xa <- fm
          ep <- fm
          em <- em + fm
          xw <- 1
          wg <- 1
          
          for(k in 1:(n-1)) {
            xw <- xw * (n + 1 - k) / k
            wg <- wg + xw
            fp <- f(a + per2 * (tn - k),...)
            xa <- xa + fp
            ep <- ep + fp * wg
            em <- em + fp * xw
          }
          wg <- per2 * n / (wg * n + xw)
          em <- wg * abs(em)
          
          if (em <= err || l >= lmax) {
            break
          }
          
          i <- i + per2 * xa
          
        }
        
        i <- i + wg * ep
        
        if (em > err) {
          err <- em
        }
      }
      
      t <- t + h
      
    if(!(t < 1)) {
      break
    }
  }
    
    if (m == 1) {
      errd <- 1 + 2 * errh
    } else {
      errd <- h * (abs(i - 2 * iback) + pq * abs(ir - 2 * irback))
    }
    h <- h * 0.5
    m <- m * 2
  if(!(errd > errh && m < mmax)) {
    break
  }
}
    
  i <- i * h
  
  if (errd > errh) {
    err <- -errd
  } else {
    err <- err * m * 0.5
  }
  
  result <- list("i" = i, "err" = err)
  
  return(result)
}

f5 <- function(x) {
  return(sin(x) / x)
}

f6 <- function(x) {
  return(cos(x) / sqrt(x))
}

result5 <- intdeo(f5, 0, 1, 1e-15)
result5$i
result5$err

result6 <- intdeo(f6, 0, 1, 1e-15)
result6$i
result6$err

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="HMJkG08ul1qO" colab_type="text" -->
# Features, Convolutions and Morphological Operations
<!-- #endregion -->

<!-- #region id="bkvjcO07s-Uk" colab_type="text" -->
https://www.tensorflow.org/api_docs/python/tf/nn/conv2d


<!-- #endregion -->

<!-- #region id="k-xq0YMbo2Co" colab_type="text" -->
$$y(n_1, n_2) = \sum_{k_1=-\infty}^{\infty} \sum_{k_2=-\infty}^{\infty} x(n_1 - k_1, n_2 - k_2)h(k_1, k_2)$$

h is

x is

$$\sum_{k_1=-\infty}^{\infty} \sum_{k_2=-\infty}^{\infty}...\sum_{k_M=-\infty}^{\infty} h(k_1,k_2,...,k_M)x(n_1-k_1,n_2-k_2,...,n_M-k_M)$$




<!-- #endregion -->

```python id="wD41AK22ml_U" colab_type="code" colab={}
from google.colab.patches import cv2_imshow
```

```python id="UHDtASdDml-m" colab_type="code" colab={}
!curl -o logo.png https://colab.research.google.com/img/colab_favicon_256px.png
import cv2
img = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)
cv2_imshow(img)
```

```python id="d5CHvLG0mi98" colab_type="code" colab={}
from IPython.display import Math, HTML

def load_mathjax_in_cell_output():
  display(HTML("<script src='https://www.gstatic.com/external_hosted/"
               "mathjax/latest/MathJax.js?config=default'></script>"))
get_ipython().events.register('pre_run_cell', load_mathjax_in_cell_output)
```

```python id="YDTQExc_mi9O" colab_type="code" outputId="10e3ff6c-9c6c-418b-ef9b-d604bf7e21be" executionInfo={"status": "ok", "timestamp": 1570005434217, "user_tz": -120, "elapsed": 527, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBKcuM3zrzWydsMeTlpor_04SYP_Fx-VlyQP2mgYpk=s64", "userId": "10136788594790905986"}} colab={"base_uri": "https://localhost:8080/", "height": 89}
# try it out:
import sympy
sympy.init_printing()
x = sympy.symbols('x')
sympy.Integral(sympy.sqrt(1 / x), x)
```

```python id="CM_U9GiBnVg9" colab_type="code" outputId="bdf8d9dd-8f74-4592-9a3d-b2d142d2fa79" executionInfo={"status": "error", "timestamp": 1570005745331, "user_tz": -120, "elapsed": 585, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBKcuM3zrzWydsMeTlpor_04SYP_Fx-VlyQP2mgYpk=s64", "userId": "10136788594790905986"}} colab={"base_uri": "https://localhost:8080/", "height": 151}

```

```python id="wu8_dOPTonvy" colab_type="code" colab={}

```

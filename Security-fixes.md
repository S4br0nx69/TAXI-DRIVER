## Correctifs de sécurité

Dans le code actuel des vulnérabilités de sécurité ont étés identifiées.

### Fix n°1 - Désérialisation non sécurisée (torch.load)

La ligne de code 265 du fichier "deep_q_learning.py contenait un vecteur d'attaque :

```python
checkpoint = torch.load(path, map_location=self.device)
```
Via torch.load() sans restriction, un utilisateur peut exécuter du code arbitraire via un .pth malveillant.

Pour corriger cette faille présente dans notre code, nous avons mis en place un weights_only=True qui empêche l'exécution de code arbitraire embarqué dans un fichier .pth malveillant.

```python
checkpoint = torch.load(path, map_location=self.device, weights_only=True)
```
d
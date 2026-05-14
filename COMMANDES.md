# Commandes Ubuntu — Digital Twin GPU Training

## ÉTAPE 1 — Ouvrir Ubuntu
Dans PowerShell ou Windows Terminal :
```
wsl -d Ubuntu
```

---

## ÉTAPE 2 — Pré-entraîner sur toutes les molécules ChEMBL (2,8M)

```bash
cd ~/Twin && source venv_tf/bin/activate && nohup python3 chembl_pretrain.py > ~/Twin/logs_chembl.txt 2>&1 & echo "PID: $!"
```

### Surveiller la progression :
```bash
tail -f ~/Twin/logs_chembl.txt
```
- Appuyer **Ctrl+C** pour quitter la surveillance (le script continue)
- Durée estimée : 30-60 min sur GPU RTX 4000

### Vérifier que le script tourne :
```bash
ps aux | grep chembl | grep -v grep
```

### Arrêter si besoin :
```bash
pkill -f chembl_pretrain.py
```

---

## ÉTAPE 3 — Entraîner le modèle QSAR sur les données CCLE réelles
(seulement après que l'étape 2 est terminée)

```bash
cd ~/Twin && source venv_tf/bin/activate && python3 fullPipeline.py --no-ppo
```

---

## ÉTAPE 4 — Optimisation DQN-SELFIES (génération de molécules)
(après l'étape 3 — génération de nouvelles molécules par reinforcement learning)

```bash
cd ~/Twin && source venv_tf/bin/activate && nohup python3 dqn_optimizer.py > ~/Twin/logs_dqn.txt 2>&1 & echo "PID: $!"
```

### Surveiller la progression :
```bash
tail -f ~/Twin/logs_dqn.txt
```
- Durée estimée : 10-20 min sur GPU RTX 4000 (2000 épisodes)
- Résultat attendu : 100% de molécules valides (garantie SELFIES)

---

## ÉTAPE 5 — Entraînement complet avec PPO (optionnel)
(alternative à l'étape 4 — génération LSTM conditionnelle)

```bash
cd ~/Twin && source venv_tf/bin/activate && python3 fullPipeline.py
```

---

## Commandes utiles

### Voir les logs en temps réel :
```bash
tail -f ~/Twin/logs_chembl.txt
```

### Voir l'utilisation du GPU :
```bash
nvidia-smi
```

### Voir les processus Python actifs :
```bash
ps aux | grep python | grep -v grep
```

### Arrêter tous les scripts Python :
```bash
pkill -f python3
```

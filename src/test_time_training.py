# base interface for TTT methods
import torch 
import torch.nn as nn
import copy
from dataclasses import dataclass


@dataclass
class TTTConfig:
    lr: float = 1e-4
    steps_per_batch: int = 10
    adaptation_task: str = "actmad" # "actmad" | "rotation"
    layers: str = "bn"              # "bn" = BN/LN seulement


# Hooks d'activation
class ActivationStats:
    """Classe utilitaire pour enregistrer les activations intermédiaires des couches BN/LN pendant l'adaptation TTT."""
    def __init__(self):
        self.hooks = []
        self.activations = {}

    def register(self, model, layer_names: list):
        for name, module in model.named_modules():
            if name in layer_names:
                h = module.register_forward_hook(self._hook(name))
                self.hooks.append(h)
    
    def _hook(self, name):
        def fn(module, inp, out):
            self.activations[name] = out
        return fn
    
    def clear(self):
        self.activations.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

def _bn_layer_names(model) -> list:
    """ Retourne les noms des toutes les BN/LN du modèle pour l'adaptation TTT."""
    return [ 
        name for name, m in model.named_modules()
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.BatchNorm1d))
    ]

# Calcul des statistiques source pour ACTMAD
def compute_source_stats(model, loader, device, save_path="source_stats.pt"):
    """Calcule et sauvegarde les statistiques d'activation (moyenne et écart-type)
    des couches BN/LN du modèle sur un ensemble de validation source, pour une utilisation ultérieure dans l'adaptation TTT ActMAD."""
    hook = ActivationStats()
    names = _bn_layer_names(model)
    hook.register(model, names)
    model.eval()
    accum = {}

    with torch.no_grad():
        for x, _ in loader:
            hook.clear()
            model(x.to(device))
            for name, act in hook.activations.items():
                dims = [i for i in range(act.ndim) if i != 1]
                mu = act.mean(dim=dims).cpu()
                sig = act.std(dim=dims, unbiased=False).cpu()
                accum.setdefault(name, {"mu": [], "sig": []})
                accum[name]["mu"].append(mu)
                accum[name]["sig"].append(sig)

    stats = {
        name: (
            torch.stack(v["mu"]).mean(0),
            torch.stack(v["sig"]).mean(0),
        )
        for name, v in accum.items()
    }
    hook.remove()
    torch.save(stats, save_path)
    print(f"[ActMAD] Statistiques sources sauvegardées dans {save_path}")
    return stats

# Perte d'adaptation ActMAD
def actmad_loss(hook: ActivationStats, source_stats: dict) -> torch.Tensor:
    """Calcule la perte d'adaptation ActMAD en comparant les statistiques d'activation du batch cible
        avec les statistiques source préalablement calculées."""
    loss = torch.zeros(1, device=next(iter(hook.activations.values())).device, requires_grad=True). squeeze()
    count = 0
    for name, act in hook.activations.items():
        if name not in source_stats:
            continue
        mu_src, sig_src = source_stats[name]
        mu_src = mu_src.to(act.device)
        sig_src = sig_src.to(act.device)

        dims = [i for i in range(act.ndim) if i != 1]
        mu_tst = act.mean(dim=dims)
        sig_tst = act.std(dim=dims, unbiased=False) + 1e-6

        loss = loss + (mu_tst - mu_src).pow(2).mean()
        loss = loss + (sig_tst - sig_src).pow(2).mean()
        count += 1
    return loss / max(count, 1)



class TTTAdapter:
    # minimal TTT adapter skeleton
    """Classe d'adaptation pour Test-Time Training (TTT) qui encapsule un modèle pré-entraîné
        et fournit des méthodes pour l'adaptation en temps réel sur des batches de données cibles non étiquetées,
        en utilisant une perte d'adaptation auto-supervisée (ex: ActMAD)."""
    def __init__(self, model, config: TTTConfig, source_stats: dict):
        # store config for adaptation settings
        self.config = config
        self.source_stats = source_stats
        self._base_model = model # modèle original non modifié

    def adapt_on_batch(self, x: torch.Tensor) -> torch.Tensor:
        # adapt model parameters using unlabeled target batch
        # todo implement self-supervised adaptation objective
        model = copy.deepcopy(self._base_model)
        model.train()

        # Geler tous les paramètres sauf les BN/LN
        for p in model.parameters():
            p.requires_grad = False
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.BatchNorm1d)):
                for p in m.parameters():
                    p.requires_grad = True

        names = _bn_layer_names(model)
        hook = ActivationStats()
        hook.register(model, names)
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.lr
        )

        # Boucle d'adaptation
        for _ in range(self.config.steps_per_batch):
            opt.zero_grad()
            hook.clear()
            model(x)
            loss = actmad_loss(hook, self.source_stats)
            loss.backward()
            opt.step()

        # Inférence avec le modèle adapté
        model.eval()
        hook.clear()
        with torch.no_grad():
            logits = model(x)

        hook.remove()
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # run inference after adaptation
        # todo replace with actual model forward pass
        self._base_model.eval()
        with torch.no_grad():
            return self._base_model(x)

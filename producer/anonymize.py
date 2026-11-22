"""Pseudonymisation des identifiants de comptes (exigence CDC : RGPD / PCI-DSS).

Un identifiant comme C1231006815 est remplacé par C + HMAC-SHA256(clé, identifiant)
tronqué. Propriétés :
  * déterministe : un même compte donne toujours le même pseudonyme, donc les agrégats
    par compte (Redis), le partitionnement Kafka par nameOrig et le graphe fonctionnent ;
  * non réversible sans la clé : la clé (ANON_KEY) reste hors du code et hors de git ;
  * la 1re lettre est conservée (C = client, M = marchand), car elle sert de feature.

C'est une pseudonymisation au sens du RGPD (art. 4.5), pas une anonymisation : avec la clé,
on peut refaire la correspondance. Tous les composants qui recherchent un compte
(graphe, embeddings, API) doivent utiliser la même clé.
"""

import hashlib
import hmac
import os

DIGEST_CHARS = 16          # 64 bits : collisions négligeables pour ~9 M de comptes


def get_key():
    key = os.getenv("ANON_KEY")
    if not key:
        raise RuntimeError("ANON_KEY absente : définir la clé de pseudonymisation "
                           "(par exemple dans infra/.env, jamais dans git)")
    return key.encode("utf-8")


def pseudonymize(account_id, key):
    """C1231006815 -> C3f9a0c2b7d1e4a58 (même entrée + même clé = même sortie)."""
    digest = hmac.new(key, account_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return account_id[:1] + digest[:DIGEST_CHARS]

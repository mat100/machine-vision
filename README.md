# Machine Vision Application

Modulární Python aplikace pro vizuální kontrolu výrobního procesu, zejména pro robota zakládajícího obrobky do CNC.

## Architektura

### Machine Vision Hub
- Centrální aplikace s webovým rozhraním
- Správa USB kamer
- Správa instancí
- Dashboard systémového stavu

### Machine Vision Instance
- Samostatná služba s vlastním REST API a web UI
- Čtyři režimy: sběr dat, třídění, trénink, produkční klasifikace
- Vlastní konfigurační soubor a pracovní prostor

## Instalace

1. **Instalace závislostí:**
```bash
pip install -r requirements.txt
```

2. **Spuštění Hub aplikace:**
```bash
python hub/main.py
```

3. **Přístup k webovému rozhraní:**
- Hub: http://localhost:8000
- Instance: http://localhost:8001 (po vytvoření)

## Struktura projektu

```
machine-vision/
├── hub/                    # Centrální aplikace
│   ├── camera/            # Správa kamer
│   ├── registry/          # Správa instancí
│   ├── templates/         # HTML šablony
│   ├── static/            # CSS, JS, obrázky
│   └── main.py           # Hlavní aplikace
├── instance/              # Šablona instance
│   ├── api/              # REST API
│   ├── ui/               # Webové rozhraní
│   ├── dataset/          # Datasety
│   ├── models/           # Trénované modely
│   └── config.yaml       # Konfigurace
├── requirements.txt       # Python závislosti
└── README.md             # Tato dokumentace

# Struktura instance dat
~/.machine-vision/instances/{instance_name}/
├── config.yaml           # Konfigurace instance
├── dataset/              # Datasety pro trénink
│   ├── unclassified/     # Nezařazené obrázky
│   └── classified/       # Zařazené obrázky podle tříd
├── models/               # Trénované modely
│   ├── metadata.json     # Metadata o modelech
│   └── *.pth            # Soubory modelů
└── production/           # Produkční data
    ├── production_stats.json  # Statistiky produkce
    └── results/          # Výsledky klasifikace
```

## Funkce

### Správa kamer
- Detekce dostupných USB kamer
- Konfigurace parametrů (rozlišení, FPS)
- Pořízení snímků

### Správa instancí
- Vytvoření nové instance
- Monitoring stavu
- Přiřazení kamer

### Machine Learning
- Sběr a třídění dat
- Trénink klasifikačních modelů
- Produkční klasifikace

## API Dokumentace

Po spuštění aplikace je dostupná na:
- http://localhost:8000/docs (Hub API)
- http://localhost:8001/docs (Instance API)

## Licence

MIT License 
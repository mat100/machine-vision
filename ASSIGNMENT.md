# Zadání projektu: Machine Vision

## 1. Úvod

Cílem projektu je vytvořit modulární Python aplikaci **Machine Vision** určenou pro vizuální kontrolu výrobního procesu, zejména pro robota zakládajícího obrobky do CNC. Aplikace bude sloužit k prevenci poškození robota nebo stroje prostřednictvím klasifikace obrazu z kamery umístěné na robotickém rameni.

## 2. Architektura systému

### 2.1 Přehled komponent

#### Machine Vision Hub

- Centrální aplikace s webovým rozhraním (HTMX + Alpine.js)
- Obsahuje:
  - Camera Manager (správa a obsluha USB kamer)
  - Správu instancí (vytváření, odstranění, monitoring)
  - Dashboard systémového stavu
- REST API pro frontend a integrace (např. Node-RED)

#### Camera Manager (součást Hubu)

- Správa všech USB kamer připojených k systému
- Funkce:
  - Detekce dostupných kamer
  - Nastavení parametrů (rozlišení, FPS, formát)
  - Snímkování (statické snapshoty)
  - Sdílení kamery mezi více instancemi (přes snapshot rozhraní)

#### Machine Vision Instance

- Samostatná služba (proces nebo kontejner) s vlastním REST API a web UI
- Obsahuje čtyři režimy:
  - Sběr dat (manuální nebo provozní)
  - Třídění dat (ruční přiřazení tříd ke snímkům)
  - Trénink modelu (volba architektury, spuštění, verzování)
  - Produkční klasifikace (ručně nebo přes API)
- Každá instance využívá vlastní konfigurační soubor (`config.yaml`), ve kterém jsou definovány parametry: název instance, připojená kamera, seznam tříd, výchozí modelová architektura, port API a cesty k datasetu a modelům

## 3. Funkční požadavky

### 3.1 Správa kamer

- Detekce dostupných USB kamer
- Možnost konfigurace každé kamery (rozlišení, FPS, formát)
- API a webové UI pro pořízení snímku

### 3.2 Správa instancí

- Možnost vytvořit novou instanci
- Odstranění nebo deaktivace instance
- Přiřazení kamer instanci (volitelné)
- Dashboard s přehledem stavu instancí
- Odkaz na webové UI každé instance

### 3.3 Sběr a třídění dat

- **Manuální režim**: uživatel klikne na "Vyfotit", vybere třídu a uloží snímek
- **Provozní režim**: snímky se ukládají automaticky, následné ruční třídění
- Dataset je organizován lokálně, odděleně pro každou instanci
- Dataset by měl být označitelný podle data nebo revize pro účely verzování
- Webový prohlížeč datasetu umožňuje přehledné ruční třídění snímků

### 3.4 Trénink modelu

- Výběr klasifikační architektury (např. ResNet18, EfficientNet)
- Možnost konfigurace parametrů tréninku (počet epoch, learning rate, atd.)
- Trénink probíhá lokálně (CPU/GPU dle dostupnosti)
- Ukládání modelů s verzováním (např. `v1.pth`, `v2.pth`)
- UI zobrazuje logy a výsledky trénování

### 3.5 Produkční klasifikace

- Možnost spustit klasifikaci:
  - Ručně přes UI
  - Automatizovaně přes API (`POST /classify`)
- Výstup ve formátu JSON obsahuje název třídy, skóre pravděpodobnosti, volitelně náhledový snímek
- Ukázkový výstup JSON:

```json
{
  "class": "OK",
  "confidence": 0.98,
  "timestamp": "2025-08-05T13:42:00Z",
  "image_path": "classified/2025-08-05_1342.png"
}
```

- Volitelné ukládání snímků z klasifikace pro budoucí analýzu nebo dotrénování

## 4. Webové rozhraní

### 4.1 Hub UI (Machine Vision Hub)

- Přehled kamer (status, ID, aktuální nastavení)
- Přehled aktivních instancí (stav, název, port, odkaz na UI)
- Možnost vytvoření nové instance
- Monitoring logů a systémových metrik

### 4.2 Instance UI (Machine Vision Instance)

- Panel režimů:
  - Sběr dat
  - Třídění
  - Trénink
  - Produkce
- Možnost definice tříd
- Výběr a konfigurace modelu
- Nahrávání datasetu a přehledný prohlížeč
- Akční tlačítka: "Vyfotit", "Přiřadit třídu", "Spustit trénink", "Spustit klasifikaci"

## 5. Použité technologie

- Python 3.11+
- FastAPI (backend pro hub i instance)
- HTMX + Alpine.js (frontend, bez nutnosti JS build toolchainu)
- OpenCV (práce s kamerou a snímkování)
- PyTorch (strojové učení a trénink modelů)
- Docker / Docker Compose (nasazení a správa služeb)
- Operační systém: Linux (Debian-based)

> **Kód musí být psán s důrazem na čitelnost, rozšiřitelnost a udržovatelnost.**
>
> - Používat návrhové vzory odpovídající Pythonu (např. dependency injection, repozitářový vzor tam, kde dává smysl)
> - Striktní oddělení backendové logiky, API, a UI šablon
> - Využívat typování pomocí `typing`, validaci pomocí `Pydantic`
> - Samostatné moduly pro správu kamer, modelů, datasetů a konfigurací
> - Jasná struktura složek a souborů, podpora dokumentace v kódu (docstringy, README)
> - Možnost psát jednotkové testy, testovatelné moduly (ne monolitická skriptová logika)

## 6. Struktura projektu

```
machine-vision/
├── hub/
│   ├── camera/
│   ├── registry/
│   ├── templates/
│   ├── static/
│   └── main.py
├── instance-template/
│   ├── api/
│   ├── ui/
│   ├── dataset/
│   ├── models/
│   └── config.yaml
├── docker-compose.yml
└── README.md
```

## 7. Minimální životaschopný produkt (MVP)

- Spustitelný Hub s web UI a REST API
- Detekce kamer a jejich obsluha (snapshot, nastavení)
- Možnost vytvořit a spravovat jednu funkční instanci
- Web UI instance podporuje:
  - Ruční sběr dat
  - Třídění datasetu
  - Spuštění tréninku
  - Produkční klasifikaci
- REST API `/classify` na straně instance
- Propojení mezi Hubem, instancí a kamerou funkční v rámci jednoho zařízení

## 8. Poznámky

- Veškerá data (snímky, modely) se ukládají lokálně
- Každá instance má vlastní pracovní prostor (dataset, modely, konfigurace)
- Kamery se využívají primárně pro pořizování statických snímků (snapshoty)
- REST API by mělo být dostupné pouze v rámci lokální sítě nebo chráněné pomocí tokenu
- Očekávaný počet paralelních instancí na zařízení: 3–5
- Cloudová synchronizace není požadována
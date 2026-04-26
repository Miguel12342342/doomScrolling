# Doomscrolling Blocker

Programa em Python que usa a webcam para detectar quando você está olhando para o celular e te pune com um rickroll automático. Quando você levantar a cabeça, o vídeo fecha sozinho.

## Como funciona

1. **Detecção de rosto** via MediaPipe (principal), dlib ou OpenCV Haar Cascades como fallback
2. **Análise de olhar** rastreia a posição da íris dentro do olho em tempo real
3. **Detecção de doomscrolling** dispara quando a íris está na parte inferior do olho (olhando para baixo)
4. **Punição** abre o `rickroll.mp4` automaticamente e exibe frases motivacionais na tela
5. **Auto-recuperação** fecha o vídeo assim que você volta a olhar para frente

## Instalação

```bash
pip install -r requirements.txt
```

Para melhor precisão, instale o MediaPipe:

```bash
pip install mediapipe
```

O modelo de detecção facial é baixado automaticamente na primeira execução.

### Fallbacks (sem MediaPipe)

**dlib** (boa precisão):
```bash
pip install dlib
# Baixe o modelo de landmarks:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

**OpenCV Haar Cascades** — funciona sem instalar nada além do `requirements.txt`.

## Uso

```bash
python main.py
```

| Estado | O que aparece na tela |
|---|---|
| Olhando para frente | "Good posture! Keep it up!" em verde |
| Olhando para baixo | Frase de incentivo em vermelho + rickroll |
| Transição | "Monitoring..." em amarelo |

Pressione `q` para sair.

### Opções

```bash
python main.py --video outra_musica.mp4   # vídeo de punição customizado
python main.py --sensitivity 0.45         # mais sensível (padrão: 0.55)
python main.py --cooldown 5               # intervalo entre frases em segundos (padrão: 3)
python main.py --threshold 3              # frames para confirmar detecção (padrão: 1)
```

## Requisitos

- Python 3.13+
- Webcam
- `opencv-python` e `numpy` (incluídos no `requirements.txt`)
- `mediapipe` (opcional, recomendado)
- `dlib` (opcional, fallback)
- VLC, Windows Media Player ou player padrão do sistema para reproduzir o vídeo

## Customização

Tudo configurável direto no `main.py`:

- **Frases** — lista `self.roasts` no `__init__`
- **Sensibilidade** — parâmetro `--sensitivity` ou `self.sensitivity`
- **Intervalo entre frases** — parâmetro `--cooldown` ou `self.roast_cooldown`
- **Vídeo de punição** — parâmetro `--video` ou `self.rickroll_path`

## Solução de problemas

**Detecção muito sensível ou insensível**
Ajuste com `--sensitivity`. Valores menores = mais sensível. O valor padrão é `0.55`.

**Vídeo não abre**
Certifique-se de que o arquivo de vídeo existe no diretório. No Windows, o programa tenta VLC, Windows Media Player e depois o player padrão do sistema.

**Vídeo não fecha automaticamente**
O programa envia um sinal de encerramento ao player. Se falhar, feche manualmente e considere instalar o VLC para melhor controle do processo.

**Câmera não abre**
Verifique se outra aplicação não está usando a webcam. No Windows, o programa usa `CAP_DSHOW` automaticamente.

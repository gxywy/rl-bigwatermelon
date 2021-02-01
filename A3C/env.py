from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait,Select
from selenium.webdriver.common.touch_actions import TouchActions
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException,NoSuchElementException,ElementNotVisibleException
import time, random, cv2
import numpy as np

url = 'http://localhost/watermelon'#"http://www.wesane.com/game/654"

def process_frame42(frame):
    frame = frame[:250, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    #frame = cv2.resize(frame, (42, 42))
    cv2.imwrite("filename.png", frame)
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame

class bigwaterlemon:
    def __init__(self):
        chrome_options = Options()
        self.width = 160
        self.height = 250
        self.last_score = 0
        self.episode_num = 0
        mobile_emulation = {
            "deviceMetrics": { "width": self.width, "height": self.height, "pixelRatio": 3.0 },
            "userAgent": "Mozilla/5.0 (Linux; Android 4.2.1; en-us; Nexus 5 Build/JOP40D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Mobile Safari/535.19"
        }
        #mobile_emulation = {'deviceName': 'Apple iPhone 5'}
        
        chrome_options = Options()
        
        chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)
        chrome_options.add_experimental_option('w3c', False)
        #chrome_options.add_argument('--no-sandbox')GameCanvas Cocos2dGameContainer
        #chrome_options.add_argument('--disable-dev-shm-usage')
        #chrome_options.add_argument('blink-settings=imagesEnabled=false') # Don't load images to ensure a high speed
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging']) # set to developer mode to avoid recognised enable-logging enable-automation

        self.browser = webdriver.Chrome(options=chrome_options)
        self.browser.get(url)
        time.sleep(10)
        self.gamecanvas = self.browser.find_element_by_id('GameCanvas')

    def get_state(self):
        state = self.gamecanvas.screenshot("test.png")
        #state = self.gamecanvas.screenshot_as_png
        state = cv2.imread("test.png")
        #state = cv2.cvtColor(np.array(state), cv2.COLOR_RGB2BGR)
        state = process_frame42(state)
        return state
    
    def act(self, x):
        x = int(x)
        if x == self.width:
            x -= 1
        actions = TouchActions(self.browser)
        actions.tap_and_hold(x, 200)
        actions.move(x, 200).perform()
        time.sleep(1)
        actions.release(x, 200).perform()

    def step(self, x):
        self.act(x)
        time.sleep(5)

        score = self.browser.execute_script("return cc.js.getClassByName('GameManager').Instance.score;")
        reward = score - self.last_score
        #print(score, self.last_score, reward)
        self.last_score = score

        done = False
        end = self.browser.execute_script("return cc.js.getClassByName('GameFunction').Instance.endOne")
        if end == 1:
            self.episode_num += 1
            self.reset()
            done = True
            self.last_score = 0
        return self.get_state(), reward, done
    
    def reset(self):
        self.browser.execute_script("cc.js.getClassByName('GameManager').Instance.RestartGame.call();")
        return self.get_state()
    
    def sample_action(self):
        return random.randint(0, self.width)


if __name__ == "__main__":
    env = bigwaterlemon()
    for i in range(300):
        r, s, d = env.step(random.randint(0, env.width))
        
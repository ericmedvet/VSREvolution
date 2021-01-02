package it.units.erallab.updater;

import com.pengrad.telegrambot.TelegramBot;
import com.pengrad.telegrambot.UpdatesListener;
import com.pengrad.telegrambot.request.SendMessage;
import com.pengrad.telegrambot.request.SendPhoto;
import okhttp3.OkHttpClient;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * @author eric on 2021/01/02 for VSREvolution
 */
public class TelegramUpdater {

  public static void main(String[] args) {
    OkHttpClient client = new OkHttpClient();
    TelegramBot bot = new TelegramBot.Builder("xxx").okHttpClient(client).build();
    bot.setUpdatesListener(updates -> {
      updates.forEach(u -> {
        System.out.println(u.message().chat().id());
        System.out.println(u.message().from());
        System.out.println(u.message().text());
        if (u.message().text().equals("img")) {
          bot.execute(new SendPhoto(
              u.message().chat().id(),
              getImage("/home/eric/Immagini/io-bn.jpg")
          ));
        } else {
          bot.execute(new SendMessage(
              u.message().chat().id(),
              String.format("I'm only a listener%n\"%s\" doesn't mean anything to me", u.message().text().replaceAll("[\\W]+", " ")))
          );
        }
      });
      return UpdatesListener.CONFIRMED_UPDATES_ALL;
    });
    if (false) {
      bot.setUpdatesListener(null);
      client.dispatcher().executorService().shutdown();
      client.connectionPool().evictAll();
    }
  }

  private static byte[] getImage(String path) {
    int l = 1024;
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (FileInputStream fis = new FileInputStream(path)) {
      byte[] buffer = new byte[l];
      int readBytes;
      while ((readBytes = fis.read(buffer)) != -1) {
        baos.write(buffer, 0, readBytes);
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
    return baos.toByteArray();
  }
}

package it.units.erallab;

import java.io.*;
import java.util.Base64;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * @author eric
 * @created 2020/08/19
 * @project VSREvolution
 */
public class Utils {

  private static final Logger L = Logger.getLogger(Utils.class.getName());

  private Utils() {
  }

  public static String safelySerialize(Serializable object) {
    try (
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(new GZIPOutputStream(baos, true))
    ) {
      oos.writeObject(object);
      oos.flush();
      oos.close();
      return Base64.getEncoder().encodeToString(baos.toByteArray());
    } catch (IOException e) {
      L.log(Level.SEVERE, String.format("Cannot serialize due to %s", e), e);
      return "";
    }
  }

  public static <T> T safelyDeserialize(String string, Class<T> tClass) {
    try (
        ByteArrayInputStream bais = new ByteArrayInputStream(Base64.getDecoder().decode(string));
        ObjectInputStream ois = new ObjectInputStream(new GZIPInputStream(bais))
    ) {
      Object o = ois.readObject();
      return (T) o;
    } catch (IOException | ClassNotFoundException e) {
      L.log(Level.SEVERE, String.format("Cannot deserialize due to %s", e), e);
      return null;
    }
  }

}

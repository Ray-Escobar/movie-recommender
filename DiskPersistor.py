import pickle
import os
from typing import List, Tuple


class DiskPersistor:
    """
    Class used to store the results of lengthy computations on disk.
    """
    __persistence_dir = "persistence"

    def perist_computation(self, computations: List[Tuple], force_update: bool = False) -> List:
        """
        Performs the provided computations and saves them to persistent storage. Next time, when this functions is called,
        the objects that exist in persistent storage are loaded instead of being computed again. If force_update is true,
        the computations are performed regardless of whether or not the objects already exist in persistent storage. The results
        are then overriden to persistent storage.

        :param computations: a list containing tuples of the computations and the name under which their result should be stored.
        :param force_update: true, if the persistent storage should be overriden, false otherwise
        :return: the list of the results corresponding to the provided computations.
        """
        results = []
        for computation, persistence_name in computations:
            if force_update or self.object_exists(persistence_name) is False:
                object = computation()
                self.save_object(persistence_name, object)
                results.append(object)
            else:
                object = self.retrieve_object(persistence_name)
                results.append(object)

        return results

    def save_object(self, name: str, object):
        """
        Saves the provided object to permanent storage.

        :param name: the name of the object
        :param object: the object to be saved to permanent storage
        """
        file_path = self.__get_file_path(name)
        self.__serialize_object(file_path, object)

    def retrieve_object(self, name: str):
        """
        Retrieve the object of the provided name from the permanent storage.

        :param name: the name of the object to be retrieved
        :return: the retrieved object
        """
        file_path = self.__get_file_path(name)
        return self.__deserialize_object(file_path)

    def object_exists(self, name: str):
        """
        Checks weather or not an object of the provided name exists in the permanent storage.

        :param name: the name of the object to be checked
        :return: true, if the object exists, false otherwise
        """
        file_path = self.__get_file_path(name)
        return os.path.exists(file_path)


    def __get_file_path(self, name: str):
        return '{}/{}.pickle'.format(self.__persistence_dir, name)

    def __serialize_object(self, file_path: str, object):
        f = open(file_path, 'wb')
        pickle.dump(object, f)
        f.close()

    def __deserialize_object(self, file_path: str):
        f = open(file_path, 'rb')
        raw_data: bytes = f.read()
        f.close()
        return pickle.loads(raw_data)